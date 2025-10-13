import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# --- Grey Wolf Optimizer (GWO) for feature selection ---
class GreyWolfOptimizer:
    def __init__(self, obj_function, num_features, population_size=20, max_iter=30):
        self.obj_function = obj_function  # function to minimize (or maximize)
        self.num_features = num_features
        self.population_size = population_size
        self.max_iter = max_iter
        
        # Initialize wolves (solutions) randomly - binary vectors indicating feature selection
        self.positions = np.random.randint(0, 2, (population_size, num_features))
        self.alpha_pos = np.zeros(num_features)
        self.alpha_score = float('-inf')
        
        self.beta_pos = np.zeros(num_features)
        self.beta_score = float('-inf')
        
        self.delta_pos = np.zeros(num_features)
        self.delta_score = float('-inf')
        
    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                # Evaluate fitness of each wolf (feature subset)
                fitness = self.obj_function(self.positions[i])
                
                # Update alpha, beta, delta wolves
                if fitness > self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness > self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness > self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            a = 2 - iteration * (2 / self.max_iter)  # linearly decreasing factor
            
            # Update positions of wolves
            for i in range(self.population_size):
                for j in range(self.num_features):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i][j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i][j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i][j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Update position with average
                    new_pos = (X1 + X2 + X3) / 3
                    
                    # Sigmoid function to map continuous to binary
                    sigmoid = 1 / (1 + np.exp(-new_pos))
                    self.positions[i][j] = 1 if sigmoid > 0.5 else 0
            
            print(f"Iteration {iteration+1}/{self.max_iter}, Best Score: {self.alpha_score:.4f}")
        
        return self.alpha_pos, self.alpha_score


# --- Objective function for feature selection ---

def feature_selection_fitness(selected_features):
    # If no features selected, assign very low fitness
    if np.count_nonzero(selected_features) == 0:
        return 0
    
    # Select features for training
    X_selected = X[:, selected_features == 1]
    
    # Use RandomForestClassifier with 5-fold CV to get accuracy
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(clf, X_selected, y, cv=5, scoring='accuracy')
    
    # Fitness = accuracy penalized by number of features (to encourage smaller subsets)
    accuracy = scores.mean()
    feature_ratio = np.sum(selected_features) / len(selected_features)
    fitness = accuracy * (1 - feature_ratio)  # balance accuracy and subset size
    
    return fitness


# --- Load example dataset (Breast Cancer) ---

data = load_breast_cancer()
X = data.data
y = data.target

num_features = X.shape[1]

# --- Run Grey Wolf Optimizer for feature selection ---

gwo = GreyWolfOptimizer(obj_function=feature_selection_fitness, num_features=num_features, population_size=20, max_iter=20)
best_features, best_score = gwo.optimize()

selected_features_indices = np.where(best_features == 1)[0]

print(f"\nBest fitness score: {best_score:.4f}")
print(f"Number of selected features: {len(selected_features_indices)}")
print(f"Selected feature indices: {selected_features_indices}")
