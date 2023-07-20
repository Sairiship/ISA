#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

// Structure to represent a transaction
typedef struct {
    double amount;
    bool is_anomaly; // flag to indicate if the transaction is an anomaly
} Transaction;

// Structure to represent a decision tree node
typedef struct DecisionTreeNode {
    int feature_index; // Index of the feature to split on
    double threshold; // Threshold value for the split
    struct DecisionTreeNode* left; // Left subtree (transactions with feature value <= threshold)
    struct DecisionTreeNode* right; // Right subtree (transactions with feature value > threshold)
} DecisionTreeNode;

// Function to train the decision tree model using labeled data
DecisionTreeNode* train_decision_tree(Transaction* transactions, int num_transactions) {
    // Check if all transactions have the same label
    bool all_same_label = true;
    bool first_label = transactions[0].is_anomaly;

    for (int i = 1; i < num_transactions; ++i) {
        if (transactions[i].is_anomaly != first_label) {
            all_same_label = false;
            break;
        }
    }

    // If all transactions have the same label, create a leaf node and return
    if (all_same_label) {
        DecisionTreeNode* leaf_node = (DecisionTreeNode*)malloc(sizeof(DecisionTreeNode));
        leaf_node->feature_index = -1;
        leaf_node->threshold = 0.0;
        leaf_node->left = NULL;
        leaf_node->right = NULL;
        return leaf_node;
    }

    // Calculate entropy of the current node
    int num_anomalies = 0;
    for (int i = 0; i < num_transactions; ++i) {
        if (transactions[i].is_anomaly) {
            num_anomalies++;
        }
    }
    double p_anomaly = (double)num_anomalies / num_transactions;
    double p_non_anomaly = 1.0 - p_anomaly;
    double entropy = -(p_anomaly * log2(p_anomaly) + p_non_anomaly * log2(p_non_anomaly));

    // Find the best split (feature and threshold) that minimizes the entropy
    int best_feature_index = -1;
    double best_threshold = 0.0;
    double min_entropy = entropy;

    for (int i = 0; i < num_transactions; ++i) {
        // Check if splitting on this feature improves entropy
        if (i > 0 && transactions[i].amount == transactions[i - 1].amount) {
            continue; // Skip duplicate feature values to avoid overfitting
        }

        // Split transactions based on the current feature value
        int num_left = 0;
        int num_right = 0;
        int num_left_anomalies = 0;
        int num_right_anomalies = 0;

        for (int j = 0; j < num_transactions; ++j) {
            if (transactions[j].amount <= transactions[i].amount) {
                num_left++;
                if (transactions[j].is_anomaly) {
                    num_left_anomalies++;
                }
            } else {
                num_right++;
                if (transactions[j].is_anomaly) {
                    num_right_anomalies++;
                }
            }
        }

        double p_left_anomaly = (double)num_left_anomalies / num_left;
        double p_left_non_anomaly = 1.0 - p_left_anomaly;
        double entropy_left = -(p_left_anomaly * log2(p_left_anomaly) + p_left_non_anomaly * log2(p_left_non_anomaly));

        double p_right_anomaly = (double)num_right_anomalies / num_right;
        double p_right_non_anomaly = 1.0 - p_right_anomaly;
        double entropy_right = -(p_right_anomaly * log2(p_right_anomaly) + p_right_non_anomaly * log2(p_right_non_anomaly));

        double weighted_entropy = (double)num_left / num_transactions * entropy_left + (double)num_right / num_transactions * entropy_right;

        if (weighted_entropy < min_entropy) {
            min_entropy = weighted_entropy;
            best_feature_index = i;
            best_threshold = transactions[i].amount;
        }
    }

    // If no split improves entropy, create a leaf node and return
    if (best_feature_index == -1) {
        DecisionTreeNode* leaf_node = (DecisionTreeNode*)malloc(sizeof(DecisionTreeNode));
        leaf_node->feature_index = -1;
        leaf_node->threshold = 0.0;
        leaf_node->left = NULL;
        leaf_node->right = NULL;
        return leaf_node;
    }

    // Split the data based on the best feature and threshold
    Transaction* left_transactions = (Transaction*)malloc(best_feature_index * sizeof(Transaction));
    Transaction* right_transactions = (Transaction*)malloc((num_transactions - best_feature_index) * sizeof(Transaction));
    int left_idx = 0;
    int right_idx = 0;

    for (int i = 0; i < num_transactions; ++i) {
        if (transactions[i].amount <= best_threshold) {
            left_transactions[left_idx++] = transactions[i];
        } else {
            right_transactions[right_idx++] = transactions[i];
        }
    }

    // Recursively create the left and right subtrees
    DecisionTreeNode* node = (DecisionTreeNode*)malloc(sizeof(DecisionTreeNode));
    node->feature_index = best_feature_index;
    node->threshold = best_threshold;
    node->left = train_decision_tree(left_transactions, best_feature_index);
    node->right = train_decision_tree(right_transactions, num_transactions - best_feature_index);

    free(left_transactions);
    free(right_transactions);

    return node;
}

// Function to predict anomalies using the trained decision tree
void predict_anomalies(Transaction* transactions, int num_transactions, DecisionTreeNode* root) {
    for (int i = 0; i < num_transactions; ++i) {
        DecisionTreeNode* current_node = root;
        while (current_node->feature_index != -1) {
            if (transactions[i].amount <= current_node->threshold) {
                current_node = current_node->left;
            } else {
                current_node = current_node->right;
            }
        }
        transactions[i].is_anomaly = current_node->threshold;
    }
}

// Function to free the memory occupied by the decision tree nodes
void free_decision_tree(DecisionTreeNode* node) {
    if (node == NULL) {
        return;
    }
    free_decision_tree(node->left);
    free_decision_tree(node->right);
    free(node);
}

int main() {
    // Sample data (you would get real data from your cryptocurrency system)
    Transaction transactions[10] = {
        {500, false},
        {1200, false},
        {800, false},
        {1500, false},
        {600, false},
        {2000, false},
        {300, false},
        {1400, false},
        {900, false},
        {1100, false}
    };

    int num_transactions = sizeof(transactions) / sizeof(transactions[0]);

    // Training the decision tree with labeled data
    DecisionTreeNode* root = train_decision_tree(transactions, num_transactions);

    // Predicting anomalies using the trained decision tree
    predict_anomalies(transactions, num_transactions, root);

    // Displaying the results
    printf("Transaction\tAmount\tIs Anomaly?\n");
    for (int i = 0; i < num_transactions; ++i) {
        printf("%d\t\t%.2f\t%s\n", i + 1, transactions[i].amount, transactions[i].is_anomaly ? "Yes" : "No");
    }

    // Free the memory occupied by the decision tree
    free_decision_tree(root);

    return 0;
}
