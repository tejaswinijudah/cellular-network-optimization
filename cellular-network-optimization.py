import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import label, gaussian_filter
import copy
import math

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
GRID_SIZE = 100
SIGNAL_THRESHOLD = 3.5  # Slightly higher threshold for stricter optimization
MAX_ITERATIONS = 12
MAX_POWER = 200

# Environment Matrix Setup
URBAN = 0
SUBURBAN = 1
RURAL = 2

# Path loss exponent (gamma) and Noise factor per environment
# Gamma reduced slightly to allow realistic propagation overlap
ENV_PARAMS = {
    URBAN:    {'gamma': 2.2, 'noise': 1.5, 'cost_mult': 2.5, 'interf_penalty': 5},
    SUBURBAN: {'gamma': 1.9, 'noise': 0.8, 'cost_mult': 1.2, 'interf_penalty': 2},
    RURAL:    {'gamma': 1.6, 'noise': 0.2, 'cost_mult': 0.8, 'interf_penalty': 0}
}

# Base Costs
COST_POWER_UNIT = 0.8
COST_SMALL_CELL = 45
COST_MACRO_CELL = 150
COST_REPOSITION = 20
COST_SECTORIZE  = 30

# ==========================================
# 2. CORE CLASSES
# ==========================================
class Tower:
    def __init__(self, x, y, power, t_type='macro'):
        self.x = int(x)
        self.y = int(y)
        self.power = power
        self.type = t_type
        # Sectorization
        self.sector_angle = None
        self.sector_width = None
        self.sector_boost = 1.0
        # Tracking history to prevent repetitive actions
        self.power_increases = 0  

class CityGrid:
    def __init__(self, size):
        self.size = size
        self.env_map = np.full((size, size), RURAL)
        self.density_map = np.zeros((size, size))
        self._generate_city()
        
    def _generate_city(self):
        center = self.size // 2
        # 1. Base Concentric Environments
        for i in range(self.size):
            for j in range(self.size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < self.size * 0.25:
                    self.env_map[i, j] = URBAN
                    self.density_map[i, j] = np.random.uniform(0.6, 0.8)
                elif dist < self.size * 0.5:
                    self.env_map[i, j] = SUBURBAN
                    self.density_map[i, j] = np.random.uniform(0.3, 0.5)
                else:
                    self.env_map[i, j] = RURAL
                    self.density_map[i, j] = np.random.uniform(0.05, 0.2)
                    
        # 2. Add Random Density Hotspots (Gaussian blobs)
        num_hotspots = np.random.randint(3, 7)
        for _ in range(num_hotspots):
            hx, hy = np.random.randint(10, self.size-10, size=2)
            strength = np.random.uniform(0.5, 1.5)
            # Create a 2D gaussian peak
            y, x = np.ogrid[-hy:self.size-hy, -hx:self.size-hx]
            mask = np.exp(-(x*x + y*y) / (2 * (self.size/10)**2))
            self.density_map += mask * strength
            
        # Smooth the final density map to make it realistic
        self.density_map = gaussian_filter(self.density_map, sigma=2)
        self.density_map = np.clip(self.density_map, 0.1, 2.0)

class NetworkSimulator:
    def __init__(self, city):
        self.city = city
        self.towers = []
        self.total_cost = 0
        self.action_log = []
        
        # Initial Setup: Place 2-5 random macro towers
        num_towers = np.random.randint(2, 6)
        for _ in range(num_towers):
            tx = np.random.randint(20, city.size-20)
            ty = np.random.randint(20, city.size-20)
            self.towers.append(Tower(tx, ty, power=np.random.uniform(60, 100), t_type='macro'))

    def compute_signal_grid(self, towers=None):
        if towers is None:
            towers = self.towers
            
        grid = np.zeros((self.city.size, self.city.size))
        y_indices, x_indices = np.indices((self.city.size, self.city.size))
        
        # Add smoothed random noise across the grid
        base_noise = gaussian_filter(np.random.normal(0, 0.3, (self.city.size, self.city.size)), sigma=1.5)
        
        for tower in towers:
            # Distance scaled down to soften the inverse distance penalty
            dist = np.sqrt((x_indices - tower.x)**2 + (y_indices - tower.y)**2)
            dist = np.maximum(dist / 4.0, 1.0) 
            
            gamma = np.vectorize(lambda e: ENV_PARAMS[e]['gamma'])(self.city.env_map)
            env_noise = np.vectorize(lambda e: ENV_PARAMS[e]['noise'])(self.city.env_map)
            
            signal = tower.power / (dist ** gamma)
            
            # Sectorization Boost
            if tower.sector_angle is not None:
                angles = np.arctan2(y_indices - tower.y, x_indices - tower.x)
                angle_diff = np.abs(np.arctan2(np.sin(angles - tower.sector_angle), np.cos(angles - tower.sector_angle)))
                in_sector = angle_diff <= (tower.sector_width / 2.0)
                signal = np.where(in_sector, signal * tower.sector_boost, signal)
                
            # Apply combined noise and clip
            signal = signal - env_noise + base_noise
            signal = np.maximum(signal, 0)
            grid = np.maximum(grid, signal) # Connect to strongest tower
            
        # Final pass filter for realistic smooth radio propagation
        return gaussian_filter(grid, sigma=0.8)

# ==========================================
# 3. OPTIMIZER MODULE
# ==========================================
class NetworkOptimizer:
    def __init__(self, simulator):
        self.sim = simulator

    def get_weak_clusters(self, signal_grid):
        weak_mask = signal_grid < SIGNAL_THRESHOLD
        labeled_array, num_features = label(weak_mask)
        clusters = []
        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_array == i)
            if len(coords) > 8: # Increased threshold to ignore micro-gaps
                cy, cx = coords.mean(axis=0)
                # Combine size AND density for prioritization
                density_sum = np.sum(self.sim.city.density_map[coords[:, 0], coords[:, 1]])
                clusters.append({
                    'coords': coords,
                    'center': (int(cx), int(cy)),
                    'size': len(coords),
                    'density_score': density_sum
                })
        # Sort by density right away
        clusters.sort(key=lambda c: c['density_score'], reverse=True)
        return clusters, weak_mask

    def evaluate_strategies(self, cluster, current_grid, force_exploration=False):
        cx, cy = cluster['center']
        env_type = self.sim.city.env_map[cy, cx]
        cost_multiplier = ENV_PARAMS[env_type]['cost_mult']
        interf_penalty  = ENV_PARAMS[env_type]['interf_penalty']
        
        nearest_tower = min(self.sim.towers, key=lambda t: (t.x - cx)**2 + (t.y - cy)**2)
        dist_to_cluster = np.sqrt((nearest_tower.x - cx)**2 + (nearest_tower.y - cy)**2)
        
        strategies = []

        def eval_virtually(virtual_towers, cost, strat_name):
            new_grid = self.sim.compute_signal_grid(virtual_towers)
            now_covered_mask = (current_grid < SIGNAL_THRESHOLD) & (new_grid >= SIGNAL_THRESHOLD)
            
            # Weighted improvement (Density driven)
            improvement = np.sum(self.sim.city.density_map[now_covered_mask])
            
            # Scoring logic with penalties
            score = improvement / cost if cost > 0 else 0
            
            # Add slight randomness to prevent rigid local minimums
            score *= np.random.uniform(0.9, 1.1)
            
            return {'name': strat_name, 'score': score, 'cost': cost, 'towers': virtual_towers, 'imp': improvement}

        # Strategy A: Increase Power (Non-linear cost to prevent spamming)
        if nearest_tower.power < MAX_POWER:
            power_boost = 25
            # Cost increases with every previous boost + interference penalty based on env
            penalty = (nearest_tower.power_increases ** 2) * 5 + interf_penalty
            cost = (power_boost * COST_POWER_UNIT * cost_multiplier) + penalty
            vtowers = copy.deepcopy(self.sim.towers)
            vt = min(vtowers, key=lambda t: (t.x - nearest_tower.x)**2 + (t.y - nearest_tower.y)**2)
            vt.power = min(vt.power + power_boost, MAX_POWER)
            vt.power_increases += 1
            strategies.append(eval_virtually(vtowers, cost, f"Boost Power (+{power_boost}W) on tower at {vt.x},{vt.y}"))

        # Strategy B: Add Small Cell
        local_density = self.sim.city.density_map[cy, cx]
        # More effective in high density (discount), penalized in rural (premium)
        density_discount = 1.5 if local_density > 1.0 else 0.8
        cost = (COST_SMALL_CELL * cost_multiplier) / density_discount
        vtowers = copy.deepcopy(self.sim.towers)
        vtowers.append(Tower(cx, cy, power=45, t_type='small'))
        strategies.append(eval_virtually(vtowers, cost, f"Deploy Small Cell at {cx},{cy}"))

        # Strategy C: Add Macro Cell (Only for massive clusters)
        if cluster['size'] > 60:
            delay_penalty = 50 # Represents time/land acquisition
            cost = (COST_MACRO_CELL * cost_multiplier) + delay_penalty
            vtowers = copy.deepcopy(self.sim.towers)
            vtowers.append(Tower(cx, cy, power=110, t_type='macro'))
            strategies.append(eval_virtually(vtowers, cost, f"Build Macro Cell at {cx},{cy}"))

        # Strategy D: Sectorization (Only if cluster is distinctly directional)
        if dist_to_cluster > 10:
            cost = COST_SECTORIZE * cost_multiplier
            vtowers = copy.deepcopy(self.sim.towers)
            vt = min(vtowers, key=lambda t: (t.x - nearest_tower.x)**2 + (t.y - nearest_tower.y)**2)
            angle = math.atan2(cy - vt.y, cx - vt.x)
            vt.sector_angle = angle
            vt.sector_width = math.pi / 2.5
            vt.sector_boost = 1.9
            strategies.append(eval_virtually(vtowers, cost, f"Sectorize tower at {vt.x},{vt.y} toward {cx},{cy}"))
            
        # Strategy E: Hybrid (Small Cell + Slight Macro Boost)
        hybrid_cost = ((COST_SMALL_CELL * cost_multiplier) / density_discount) + (10 * COST_POWER_UNIT)
        vtowers = copy.deepcopy(self.sim.towers)
        vtowers.append(Tower(cx, cy, power=35, t_type='small'))
        vt = min(vtowers, key=lambda t: (t.x - nearest_tower.x)**2 + (t.y - nearest_tower.y)**2)
        vt.power = min(vt.power + 10, MAX_POWER)
        strategies.append(eval_virtually(vtowers, hybrid_cost, f"Hybrid: Small Cell at {cx},{cy} + Macro Boost"))

        if not strategies: return None
        
        # Sort strategies by score
        strategies.sort(key=lambda x: x['score'], reverse=True)
        
        # If stuck in a local optimum, pick the 2nd best strategy occasionally
        if force_exploration and len(strategies) > 1:
            return strategies[1]
        return strategies[0]

    def optimize(self):
        print("Starting Iterative Optimization...")
        initial_grid = self.sim.compute_signal_grid()
        _, initial_weak_mask = self.get_weak_clusters(initial_grid)
        
        stagnation_counter = 0
        last_weak_cells = np.sum(initial_weak_mask)
        
        for iteration in range(MAX_ITERATIONS):
            current_grid = self.sim.compute_signal_grid()
            clusters, _ = self.get_weak_clusters(current_grid)
            
            if not clusters:
                print("Network perfectly optimized. Zero critical weak zones.")
                break
                
            target_cluster = clusters[0]
            force_explore = (stagnation_counter >= 2)
            
            best_strat = self.evaluate_strategies(target_cluster, current_grid, force_exploration=force_explore)
            
            if best_strat and best_strat['score'] > 0:
                self.sim.towers = best_strat['towers']
                self.sim.total_cost += best_strat['cost']
                self.sim.action_log.append(best_strat['name'])
                
                # Check for stagnation (Improvement of less than 20 cells)
                current_weak_cells = np.sum(current_grid < SIGNAL_THRESHOLD)
                if (last_weak_cells - current_weak_cells) < 20:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                last_weak_cells = current_weak_cells
                
                flag = "[EXPLORATION]" if force_explore else ""
                print(f"Iter {iteration+1:02d}: {flag} Applied -> {best_strat['name']} (Cost: {best_strat['cost']:.1f}, Score: {best_strat['score']:.2f})")
            else:
                print(f"Iter {iteration+1:02d}: No viable strategies for top cluster. Halting.")
                break
                
        final_grid = self.sim.compute_signal_grid()
        _, final_weak_mask = self.get_weak_clusters(final_grid)
        return initial_grid, initial_weak_mask, final_grid, final_weak_mask

# ==========================================
# 4. VISUALIZATION MODULE
# ==========================================
def plot_results(city, initial_grid, init_weak, final_grid, final_weak, init_towers, final_towers):
    fig = plt.figure(figsize=(16, 10))
    # Adjust spacing for clean professional look
    plt.subplots_adjust(wspace=0.25, hspace=0.35, top=0.9)
    
    # Custom Legend Elements
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='white', markersize=12, markeredgecolor='black', label='Macro Cell'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='Small Cell'),
        Line2D([0], [0], color='yellow', lw=2, label='Sector Direction')
    ]

    def plot_towers(ax, towers):
        for t in towers:
            marker = '^' if t.type == 'macro' else 'o'
            color = 'white' if t.type == 'macro' else 'cyan'
            size = 180 if t.type == 'macro' else 90
            ax.scatter(t.x, t.y, c=color, marker=marker, s=size, edgecolors='black', zorder=5)
            if t.sector_angle is not None:
                dx = 8 * math.cos(t.sector_angle)
                dy = 8 * math.sin(t.sector_angle)
                ax.arrow(t.x, t.y, dx, dy, color='yellow', width=0.8, zorder=4)

    # 1. Initial Coverage (Plasma with Contours)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(initial_grid, cmap='plasma', vmin=0, vmax=12)
    ax1.contour(initial_grid, levels=[SIGNAL_THRESHOLD], colors='red', linestyles='dashed', alpha=0.6)
    plot_towers(ax1, init_towers)
    ax1.set_title("** Initial Coverage **", fontweight='bold')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Density Map & Initial Weak Zones
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(city.density_map, cmap='magma')
    ax2.imshow(init_weak, cmap='Reds', alpha=0.45) 
    plot_towers(ax2, init_towers)
    ax2.set_title("** Density & Weak Zones (Red) **", fontweight='bold')

    # 3. Final Coverage (Plasma with Contours)
    ax3 = plt.subplot(2, 3, 4)
    im3 = ax3.imshow(final_grid, cmap='plasma', vmin=0, vmax=12)
    ax3.contour(final_grid, levels=[SIGNAL_THRESHOLD], colors='red', linestyles='dashed', alpha=0.6)
    plot_towers(ax3, final_towers)
    ax3.set_title("** Optimized Coverage **", fontweight='bold')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Final Weak Zones 
    ax4 = plt.subplot(2, 3, 5)
    ax4.imshow(city.density_map, cmap='magma')
    ax4.imshow(final_weak, cmap='Reds', alpha=0.45)
    plot_towers(ax4, final_towers)
    ax4.set_title("** Remaining Weak Zones **", fontweight='bold')

    # 5. Improvement Delta Map (Diverging Colormap)
    ax5 = plt.subplot(1, 3, 3) # Span the right side
    diff_grid = final_grid - initial_grid
    # Diverging colormap: Red is negative, Green is positive improvement
    im5 = ax5.imshow(diff_grid, cmap='RdYlGn', vmin=-5, vmax=5)
    plot_towers(ax5, final_towers)
    ax5.set_title("** Network Improvement Map **\n(Green = Gained Signal)", fontweight='bold')
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Add Global Legend to figure
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, frameon=True)

    plt.show()

# ==========================================
# 5. EXECUTION & DASHBOARD
# ==========================================
if __name__ == "__main__":
    
    # Initialize components
    my_city = CityGrid(GRID_SIZE)
    sim = NetworkSimulator(my_city)
    optimizer = NetworkOptimizer(sim)
    
    initial_towers = copy.deepcopy(sim.towers)
    
    # Run Optimization
    init_g, init_w, final_g, final_w = optimizer.optimize()
    
    # Dashboard Metrics Calculations
    init_weak_count = np.sum(init_w)
    final_weak_count = np.sum(final_w)
    cells_improved = init_weak_count - final_weak_count
    improvement_pct = (cells_improved / init_weak_count * 100) if init_weak_count > 0 else 0
    
    avg_sig_init = np.mean(init_g)
    avg_sig_final = np.mean(final_g)
    
    added_small_cells = sum(1 for t in sim.towers if t.type == 'small')
    added_macros = sum(1 for t in sim.towers if t.type == 'macro') - sum(1 for t in initial_towers if t.type == 'macro')
    cost_efficiency = cells_improved / sim.total_cost if sim.total_cost > 0 else 0

    print("\n" + "="*45)
    print(" 📡 NETWORK OPTIMIZATION DASHBOARD 📡")
    print("="*45)
    print(f"▶ Initial Towers      : {len(initial_towers)} Macros")
    print(f"▶ Initial Weak Cells  : {init_weak_count}")
    print(f"▶ Final Weak Cells    : {final_weak_count}")
    print(f"▶ Weak Zone Reduction : {improvement_pct:.2f}%")
    print("-" * 45)
    print(f"▶ Avg Signal Start    : {avg_sig_init:.2f} dBm")
    print(f"▶ Avg Signal Final    : {avg_sig_final:.2f} dBm")
    print(f"▶ Network Total Cost  : ${sim.total_cost:.2f}")
    print(f"▶ Cost Efficiency     : {cost_efficiency:.3f} cells/dollar")
    print("-" * 45)
    print(f"▶ Small Cells Added   : {added_small_cells}")
    print(f"▶ Macros Added        : {added_macros}")
    print(f"▶ Total Strategy Steps: {len(sim.action_log)}")
    print("="*45)
    
    # Visualize
    plot_results(my_city, init_g, init_w, final_g, final_w, initial_towers, sim.towers)