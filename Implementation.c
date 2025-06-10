#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

// Structure to represent a financial transaction
typedef struct {
    double amount;      // Transaction amount in currency
    bool is_anomaly;    // True if transaction is fraudulent/anomalous
} Transaction;

// Node structure for the decision tree
typedef struct DecisionTreeNode {
    int feature_index;      // Index of feature to split on (-1 for leaf nodes)
    double threshold;       // Threshold value for splitting 
    struct DecisionTreeNode* left;   // Left child (values <= threshold)
    struct DecisionTreeNode* right;  // Right child (values > threshold)
} DecisionTreeNode;

/**
 * Safe logarithm base 2 function to avoid log(0) errors
 * Returns 0 for x <= 0, log2(x) otherwise
 */
double safe_log2(double x) {
    return x > 0 ? log2(x) : 0.0;
}

/**
 * Comparison function for qsort to sort transactions by amount
 * Used to find optimal split points in the decision tree
 */
int compare_transactions(const void* a, const void* b) {
    double diff = ((Transaction*)a)->amount - ((Transaction*)b)->amount;
    return (diff > 0) - (diff < 0);  // Returns -1, 0, or 1
}

 //Recursively builds a decision tree using entropy-based splits

DecisionTreeNode* train_decision_tree(Transaction* transactions, int num_transactions) {
    // Base case: Check if all transactions have the same label
    bool first_label = transactions[0].is_anomaly;
    bool all_same_label = true;
    
    for (int i = 1; i < num_transactions; ++i) {
        if (transactions[i].is_anomaly != first_label) {
            all_same_label = false;
            break;
        }
    }
    
    // Create leaf node if all labels are same or insufficient data
    if (all_same_label || num_transactions <= 1) {
        DecisionTreeNode* leaf_node = (DecisionTreeNode*)malloc(sizeof(DecisionTreeNode));
        leaf_node->feature_index = -1;  // -1 indicates leaf node
        leaf_node->threshold = first_label ? 1.0 : 0.0;  // Store prediction
        leaf_node->left = NULL;
        leaf_node->right = NULL;
        return leaf_node;
    }

    // Sort transactions by amount to find potential split points
    qsort(transactions, num_transactions, sizeof(Transaction), compare_transactions);

    // Calculate base entropy (entropy before any split)
    int num_anomalies = 0;
    for (int i = 0; i < num_transactions; ++i) {
        if (transactions[i].is_anomaly) num_anomalies++;
    }
    
    double p_anomaly = (double)num_anomalies / num_transactions;
    double p_normal = 1.0 - p_anomaly;
    double base_entropy = -(p_anomaly * safe_log2(p_anomaly) + p_normal * safe_log2(p_normal));

    // Find the best split point by minimizing weighted entropy
    int best_index = -1;
    double best_threshold = 0.0;
    double best_entropy = base_entropy;

    // Try all possible split points between consecutive different amounts
    for (int i = 1; i < num_transactions; ++i) {
        // Skip if amounts are the same to indicate no meaningful split 
        if (transactions[i].amount == transactions[i - 1].amount) continue;
        
        // Calculate threshold as midpoint between consecutive amounts
        double threshold = (transactions[i].amount + transactions[i - 1].amount) / 2.0;

        // Count samples and anomalies in left and right splits
        int left_count = 0, left_anomalies = 0;
        int right_count = 0, right_anomalies = 0;

        for (int j = 0; j < num_transactions; ++j) {
            if (transactions[j].amount <= threshold) {
                left_count++;
                if (transactions[j].is_anomaly) left_anomalies++;
            } else {
                right_count++;
                if (transactions[j].is_anomaly) right_anomalies++;
            }
        }

        // Calculate entropy for left and right splits
        double pl = (double)left_anomalies / left_count;
        double pr = (double)right_anomalies / right_count;

        double entropy_left = -(pl * safe_log2(pl) + (1 - pl) * safe_log2(1 - pl));
        double entropy_right = -(pr * safe_log2(pr) + (1 - pr) * safe_log2(1 - pr));

        // Calculate weighted average entropy after split
        double weighted_entropy = (left_count * entropy_left + right_count * entropy_right) / num_transactions;

        // Update best split if this one reduces entropy more
        if (weighted_entropy < best_entropy) {
            best_entropy = weighted_entropy;
            best_threshold = threshold;
            best_index = i;
        }
    }

    // If no good split found, create leaf with majority class
    if (best_index == -1) {
        DecisionTreeNode* leaf = (DecisionTreeNode*)malloc(sizeof(DecisionTreeNode));
        leaf->feature_index = -1;
        leaf->threshold = p_anomaly >= 0.5 ? 1.0 : 0.0;  // Majority vote
        leaf->left = NULL;
        leaf->right = NULL;
        return leaf;
    }

    // Split the data into left and right subsets based on best threshold
    int left_size = 0, right_size = 0;
    for (int i = 0; i < num_transactions; ++i) {
        if (transactions[i].amount <= best_threshold) left_size++;
        else right_size++;
    }

    // Allocate memory for left and right subsets
    Transaction* left_data = (Transaction*)malloc(sizeof(Transaction) * left_size);
    Transaction* right_data = (Transaction*)malloc(sizeof(Transaction) * right_size);

    // Populate left and right subsets
    int li = 0, ri = 0;
    for (int i = 0; i < num_transactions; ++i) {
        if (transactions[i].amount <= best_threshold)
            left_data[li++] = transactions[i];
        else
            right_data[ri++] = transactions[i];
    }

    // Create internal node and recursively build left and right subtrees
    DecisionTreeNode* node = (DecisionTreeNode*)malloc(sizeof(DecisionTreeNode));
    node->feature_index = 0;  // Using transaction amount as feature
    node->threshold = best_threshold;
    node->left = train_decision_tree(left_data, left_size);   // Recursive call for left
    node->right = train_decision_tree(right_data, right_size); // Recursive call for right

    // Clean up allocated memory for subsets
    free(left_data);
    free(right_data);

    return node;
}

// Makes a prediction for a new transaction using the trained decision tree

bool predict(DecisionTreeNode* root, double amount) {
    // Traverse the tree until reaching a leaf node
    while (root->feature_index != -1) {
        if (amount <= root->threshold)
            root = root->left;   // Go left if amount <= threshold
        else
            root = root->right;  // Go right if amount > threshold
    }
    
    // Return prediction stored in leaf node (1.0 = anomaly, 0.0 = normal)
    return root->threshold == 1.0;
}


// Main function demonstrating the fraud detection system

int main() {
    // Sample training data: normal transactions (50-65) and anomalous ones (1000+)
    Transaction data[] = {
        {50.0, false},   // Normal transaction
        {60.0, false},   // Normal transaction
        {1000.0, true},  // Fraudulent transaction
        {1200.0, true},  // Fraudulent transaction
        {55.0, false},   // Normal transaction
        {65.0, false},   // Normal transaction
        {1100.0, true},  // Fraudulent transaction
        {52.0, false}    // Normal transaction
    };
    int n = sizeof(data) / sizeof(data[0]);

    // Training the Decison tree
    printf("Training decision tree with %d transactions...\n", n);
    DecisionTreeNode* tree = train_decision_tree(data, n);
    printf("Training completed!\n\n");

    // Take input from user 
    double test_amount;
    printf("Enter transaction amount: ");
    scanf("%lf", &test_amount);

    // Make prediction using trained tree
    bool result = predict(tree, test_amount);
    printf("Prediction: %s\n", result ? "Anomaly (Potential Fraud)" : "Normal Transaction");
 
    return 0;
}
