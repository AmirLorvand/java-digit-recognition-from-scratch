/**
 * This program implements a Support Vector Machine (SVM) for multi-class digit recognition
 * using handwritten digit datasets. It employs a centroid kernel-based feature transformation
 * combined with linear SVM training for classification.
 *
 * Key Features:
 * - Reads datasets from CSV files and extracts features (input pixels) and labels (output classes).
 * - Performs centroid-based feature transformation, projecting 64D features to 74D by 
 *   appending distances to class centroids.
 * - Implements a linear SVM model for binary classification using hyperparameter tuning.
 * - Uses a one-vs-all strategy to handle multi-class classification (10 classes: 0-9).
 * - Performs 2-fold cross-validation for robust evaluation.
 * - Includes a grid search to optimise hyperparameters:
 *   * Learning rate
 *   * Number of epochs
 *   * Regularisation strength
 *   * Violation penalty
 *   * Regularisation factor
 *   * Margin threshold
 * - Outputs the best combination of hyperparameters and corresponding accuracy.
 *
 * Program Execution Notes:
 * - Running a full grid search with all hyperparameter combinations is computationally intensive.
 * - Adjust the hyperparameter arrays in the code to reduce the search space and execution time.
 * - Example hyperparameters for optimal performance:
 *   * Learning rate = {1e-6}
 *   * Epochs = {1500}
 *   * Regularisation strength = {5.0}
 *   * Violation penalty = {2.0}
 *   * Regularisation factor = {0.001}
 *   * Margin threshold = {2.0}
 *
 * How to Use:
 * 1. Prepare two datasets in CSV format with the last column as labels.
 * 2. Place the datasets in the working directory and update the file paths.
 * 3. Run the program to pre-process data, train the model, and perform grid search.
 * 4. View the results, including the best hyperparameter configuration and accuracy.
 *
 * Example:
 * Input: "dataSet1.csv", "dataSet2.csv"
 * Output: Best hyperparameters and classification accuracy after grid search.
 *
 * Note:
 * - Ensure the dataset follows the required format (features and labels).
 * - The nested loops in the grid search are necessary for exhaustive hyperparameter evaluation.
 *
 * Author: Amir Lorvand
 * Date: 12 Dec 2024
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class SVM_centroid_kernel {

	// constants for parameters and paths
	static final int TRUE_CLASS = 1; // value for true class in one-vs-all labels
	static final int FALSE_CLASS = -1; // value for other classes in one-vs-all labels
	static final int NUMBER_OF_CLASSES = 10;
	static final String FILE_PATH_1 = "dataSet1.csv";
	static final String FILE_PATH_2 = "dataSet2.csv";
	static final double PERCENTAGE_MULTIPLIER = 100;

	// hyperparameters grids
	static final double[] LEARNING_RATES = { 1e-6, 6e-7, 1e-7, 5e-7, 1e-5 };
	static final int[] EPOCHS = { 250, 500, 1000, 1500 };
	static final double[] REGULARISATION_STRENGTHS = { 2, 1, 7, 5, 10, 0.5 };
	static final double[] VIOLATION_PENALTIES = { 0.5, 1, 1.5, 2, 2.5, 3 };
	static final double[] REGULARISATION_FACTORS = { 0.0001, 0.001, 0.01, 1 };
	static final double[] MARGIN_THRESHOLDS = { 0.5, 1, 1.5, 2, 2.5 };

	// reads a dataset from a CSV file and converts it to a 2D integer array
	static int[][] read_csv(String file_path) {
		ArrayList<int[]> data_list = new ArrayList<>();
		try (Scanner scanner = new Scanner(new File(file_path))) {
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] values = line.split(",");
				int[] int_values = new int[values.length];
				// converts values to integers
				for (int column_index = 0; column_index < values.length; column_index++) {
					int_values[column_index] = Integer.parseInt(values[column_index]);
				}
				data_list.add(int_values);
			}
		} catch (FileNotFoundException exception) {
			exception.printStackTrace();
		}
		// convert list to 2D array
		int[][] data_array = new int[data_list.size()][];
		for (int row_index = 0; row_index < data_list.size(); row_index++) {
			data_array[row_index] = data_list.get(row_index);
		}
		return data_array;
	}

	// extracts input features (first 64 columns) from the dataset
	static int[][] extract_features(int[][] dataset) {
		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;
		int[][] features = new int[number_row][number_of_features];

		// copy feature values from dataset
		for (int image_index = 0; image_index < features.length; image_index++) {
			for (int pixel_index = 0; pixel_index < features[0].length; pixel_index++) {
				features[image_index][pixel_index] = dataset[image_index][pixel_index];
			}
		}
		return features;
	}

	// extracts labels (last column) from the dataset
	static int[] extract_labels(int[][] dataset) {
		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;
		int[] labels = new int[number_row];

		// extract label values from dataset
		for (int image_index = 0; image_index < labels.length; image_index++) {
			labels[image_index] = dataset[image_index][number_of_features];
		}
		return labels;
	}

	// converts a 2D integer array to a 2D double array
	static double[][] convert_to_double_array(int[][] array) {
		double[][] updated_array = new double[array.length][array[0].length];
		for (int row = 0; row < array.length; row++) {
			for (int column = 0; column < array[0].length; column++) {
				updated_array[row][column] = (double) array[row][column];
			}
		}
		return updated_array;
	}

	// a function to normalise dataset (divide each pixel by the largest value)
	static double[][] normalise(double[][] dataset) {

		// find the largest value in the dataset
		double max_value = Double.MIN_VALUE;
		for (int row = 0; row < dataset.length; row++) {
			for (int column = 0; column < dataset[0].length; column++) {
				if (dataset[row][column] > max_value) {
					max_value = dataset[row][column];
				}
			}
		}

		// normalise dataset by the largest value
		double[][] normalised_dataset = new double[dataset.length][dataset[0].length];
		for (int row = 0; row < dataset.length; row++) {
			for (int column = 0; column < dataset[0].length; column++) {
				normalised_dataset[row][column] = dataset[row][column] / max_value;
			}
		}
		return normalised_dataset;
	}

	// initialises centroids to zero for all categories
	static double[][] initialise_centroids(int number_of_categories, int number_of_features) {

		double[][] centroids = new double[number_of_categories][number_of_features];

		for (int category = 0; category < number_of_categories; category++) {
			for (int feature_index = 0; feature_index < number_of_features; feature_index++) {
				centroids[category][feature_index] = 0.0;
			}
		}
		return centroids;
	}

	// a function to count how many data points belong to each category (counter)
	static int[] count_data_points(int[] labels, int number_of_categories) {

		int[] counts = new int[number_of_categories];
		// count occurrences of each category
		for (int label : labels) {
			counts[label]++;
		}
		return counts;
	}

	// accumulates the sum of feature values for each category
	static void sum_features_per_category(double[][] dataset, int[] labels, double[][] centroids) {

		for (int data_point_index = 0; data_point_index < dataset.length; data_point_index++) {
			int category = labels[data_point_index]; // determine category
			for (int feature_index = 0; feature_index < dataset[data_point_index].length; feature_index++) {
				centroids[category][feature_index] += dataset[data_point_index][feature_index];
			}
		}
	}

	// averages the feature sums to compute centroids
	static void average_centroids(double[][] centroids, int[] counts) {

		for (int category = 0; category < centroids.length; category++) {
			for (int feature_index = 0; feature_index < centroids[category].length; feature_index++) {
				if (counts[category] > 0) {
					centroids[category][feature_index] /= counts[category]; // compute average
				}
			}
		}
	}

	// calculates the centroids of each category
	static double[][] calculate_centroid_categories(double[][] dataset, int[] labels) {

		double[][] centroids = initialise_centroids(NUMBER_OF_CLASSES, dataset[0].length); // initialise centroids
		int[] counts = count_data_points(labels, NUMBER_OF_CLASSES); // count data points per category
		sum_features_per_category(dataset, labels, centroids); // accumulate feature values
		average_centroids(centroids, counts); // average feature values

		return centroids;
	}

	// calculates the euclidean distance between a data point and a centroid
	static double calculate_euclidean_distance(double[] point, double[] centroid) {
		double sum = 0;
		for (int dimension = 0; dimension < point.length; dimension++) {
			sum += Math.pow(point[dimension] - centroid[dimension], 2);
		}
		return Math.sqrt(sum);
	}

	// projects a 64D arrays to a 74D array by appending the distance to centroids
	static double[][] transform_features(double[][] features, double[][] centroids) {
		final int source_start_index = 0; // starting index in the source array
		final int dest_start_index = 0; // starting index in the destination array

		double[][] transformed_features = new double[features.length][features[0].length + centroids.length];
		// for each sample, append distances to centroids
		for (int sample_index = 0; sample_index < features.length; sample_index++) {
			// copy original features
			System.arraycopy(features[sample_index], source_start_index, transformed_features[sample_index],
					dest_start_index, features[0].length);

			// append distances to centroids
			for (int centroid_index = 0; centroid_index < centroids.length; centroid_index++) {
				transformed_features[sample_index][features[0].length + centroid_index] = calculate_euclidean_distance(
						features[sample_index], centroids[centroid_index]);
			}
		}
		return transformed_features;
	}

	// converts multi-class labels into binary labels for a specific positive class
	static int[] convert_labels(int[] labels, int positive_class) {
		int[] binary_labels = new int[labels.length];
		// assign TRUE_CLASS (1) for positive class, FALSE_CLASS (-1) otherwise
		for (int label_index = 0; label_index < labels.length; label_index++) {
			binary_labels[label_index] = (labels[label_index] == positive_class) ? TRUE_CLASS : FALSE_CLASS;
		}
		return binary_labels;
	}

	// a function to calculate the dot product of two vectors
	static double dot_product(double[] vector_1, double[] vector_2) {
		double result = 0;
		for (int index = 0; index < vector_1.length; index++) {
			result += vector_1[index] * vector_2[index];
		}
		return result;
	}

	// trains a linear SVM for binary classification
	static double[] train_linear_svm(double[][] features, int[] labels, double learning_rate, int epochs,
			double regularisation_strength, double violation_penalty, double regularisation_factor,
			double margin_threshold) {
		// initialise weights and bias
		double[] weights = new double[features[0].length];
		double bias = 0;
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int current_sample = 0; current_sample < features.length; current_sample++) {
				double margin = labels[current_sample] * (dot_product(weights, features[current_sample]) + bias); // compute
																													// margin
				if (margin < margin_threshold) {
					// update weights and bias for margin violations
					for (int feature_index = 0; feature_index < weights.length; feature_index++) {
						weights[feature_index] += learning_rate
								* (regularisation_strength * labels[current_sample] * features[current_sample][feature_index]
										- violation_penalty * weights[feature_index]);
					}
					bias += regularisation_strength * learning_rate * labels[current_sample];
				} else {
					// apply regularisation to weights
					for (int feature_index = 0; feature_index < weights.length; feature_index++) {
						weights[feature_index] -= learning_rate * regularisation_factor * weights[feature_index];
					}
				}
			}
		}
		final int weights_source_start_index = 0; // start index in weights array
		final int model_dest_start_index = 0; // start index in model array
		double[] model = new double[weights.length + 1]; // model includes weights and bias
		System.arraycopy(weights, weights_source_start_index, model, model_dest_start_index, weights.length);
		model[weights.length] = bias; // add bias to model
		return model; // return trained model
	}

	// predicts using the trained SVM model
	static int[] predict(double[][] features, double[] model) {

		double[] weights = new double[features[0].length];

		final int model_weights_start_index = 0; // start index in model array for weights
		final int weights_dest_start_index = 0; // start index in weights array

		// copy the weight values from the model array into the weights array
		System.arraycopy(model, model_weights_start_index, weights, weights_dest_start_index, features[0].length);
		double bias = model[features[0].length]; // extract bias

		int[] predictions = new int[features.length];
		// predict for each sample
		for (int sample_index = 0; sample_index < features.length; sample_index++) {
			double result = dot_product(weights, features[sample_index]) + bias; // compute prediction
			predictions[sample_index] = (result >= 0) ? TRUE_CLASS : FALSE_CLASS; // classify based on the result
		}
		return predictions;
	}

	// calculates the accuracy of predictions
	static double calculate_accuracy(int[] true_labels, int[] predicted_labels) {

		int correct = 0;
		// compare true labels and predicted labels
		for (int label_index = 0; label_index < true_labels.length; label_index++) {
			if (true_labels[label_index] == predicted_labels[label_index]) {
				correct++;
			}
		}
		return (double) correct / true_labels.length * PERCENTAGE_MULTIPLIER; // return accuracy as percentage
	}

	// a function to evaluate one fold of training and testing
	static double evaluate_fold(double[][] training_features, int[] training_labels, double[][] testing_features,
			int[] testing_labels, int class_label, double learning_rate, int epoch, double regularisation_strength,
			double violation_penalty, double regularisation_factor, double margin_threshold) {

		// convert labels to binary for the given class
		int[] binary_training_labels = convert_labels(training_labels, class_label);
		int[] binary_testing_labels = convert_labels(testing_labels, class_label);

		// train SVM for the current class
		double[] model = train_linear_svm(training_features, binary_training_labels, learning_rate, epoch,
				regularisation_strength, violation_penalty, regularisation_factor, margin_threshold);

		// predict the labels for the testing dataset using the trained model
		int[] predictions = predict(testing_features, model);

		// calculate and return the accuracy of predictions compared to the true binary
		// labels
		return calculate_accuracy(binary_testing_labels, predictions);
	}

	// a function to pre-process datasets by reading, extracting features and labels,
	// calculating centroids, and transforming features.
	static double[][][] pre_process_datasets(String file_path_1, String file_path_2) {
		// read datasets
		int[][] dataset_1 = read_csv(file_path_1);
		int[][] dataset_2 = read_csv(file_path_2);

		double[][] features_1 = convert_to_double_array(extract_features(dataset_1)); // extract features and convert
																						// them to double
		int[] labels_1 = extract_labels(dataset_1); // extract labels
		double[][] centroids_1 = calculate_centroid_categories(features_1, labels_1); // calculate centroid for each
																						// class in the labels
		double[][] transformed_features1 = transform_features(features_1, centroids_1); // project features to 74D

		double[][] features_2 = convert_to_double_array(extract_features(dataset_2));
		int[] labels_2 = extract_labels(dataset_2);
		double[][] centroids_2 = calculate_centroid_categories(features_2, labels_2);
		double[][] transformed_features_2 = transform_features(features_2, centroids_2);

		// return the transformed features and centroids for both datasets
		return new double[][][] { transformed_features1, transformed_features_2, centroids_1, centroids_2 };
	}

	// a function to evaluate all classes by training and testing the model for each
	// class
	static double[] evaluate_all_classes(double[][] training_features, int[] training_labels,
			double[][] testing_features, int[] testing_labels, double learning_rate, int epoch,
			double regularisation_strength, double violation_penalty, double regularisation_factor,
			double margin_threshold) {
		int number_of_folds = 2;
		double[] total_accuracy_per_class = new double[NUMBER_OF_CLASSES]; // store accuracy of each class

		// loop through each class
		for (int class_label = 0; class_label < NUMBER_OF_CLASSES; class_label++) {
			// evaluate model accuracy for the first fold
			double accuracy_fold_1 = evaluate_fold(training_features, training_labels, testing_features, testing_labels,
					class_label, learning_rate, epoch, regularisation_strength, violation_penalty,
					regularisation_factor, margin_threshold);
			// evaluate model accuracy for the second fold
			double accuracy_fold_2 = evaluate_fold(testing_features, testing_labels, training_features, training_labels,
					class_label, learning_rate, epoch, regularisation_strength, violation_penalty,
					regularisation_factor, margin_threshold);
			total_accuracy_per_class[class_label] = (accuracy_fold_1 + accuracy_fold_2) / number_of_folds; // calculate
																											// accuracy
																											// of both
																											// folds
		}
		return total_accuracy_per_class;
	}

	// a function to calculate the overall accuracy across all classes
	static double calculate_overall_accuracy(double[] class_accuracies) {
		double overall_accuracy = 0;
		// sum up accuracies of all classes
		for (double class_accuracy : class_accuracies) {
			overall_accuracy += class_accuracy;
		}
		return overall_accuracy / NUMBER_OF_CLASSES;
	}

	// prints the final result (best hyperparameters)
	static void print_best_parameters(double best_learning_rate, int best_epoch, double best_regularisation_strength,
			double best_violation_penalty, double best_regularisation_factor, double best_margin_threshold,
			double best_accuracy) {
		System.out.println("Best LR = " + best_learning_rate);
		System.out.println("Best Number of Epoch = " + best_epoch);
		System.out.println("Best Regularisation Strength = " + best_regularisation_strength);
		System.out.println("Best Regularisation Factor = " + best_regularisation_factor);
		System.out.println("Best Violation Penalty = " + best_violation_penalty);
		System.out.println("Best Margin Threshold = " + best_margin_threshold);
		System.out.println("Best Accuracy = " + best_accuracy);
	}

	// a function to update the best parameters if the overall accuracy improves
	static double update_best_parameters(double current_accuracy, double overall_accuracy, double learning_rate,
			int epoch, double regularisation_strength, double violation_penalty, double regularisation_factor,
			double margin_threshold, double[] best_params) {
		if (overall_accuracy > current_accuracy) {
			best_params[0] = learning_rate;
			best_params[1] = epoch;
			best_params[2] = regularisation_strength;
			best_params[3] = violation_penalty;
			best_params[4] = regularisation_factor;
			best_params[5] = margin_threshold;
			return overall_accuracy;
		}
		return current_accuracy;
	}

	public static void main(String[] args) throws Exception {
		// NOTE: The nested loops are necessary for an exhaustive search and cannot be
		// shortened without
		// compromising the thoroughness of the grid search process.

		// extract training and testing features and labels from the pre-processed
		// datasets
		double[][][] datasets = pre_process_datasets(FILE_PATH_1, FILE_PATH_2);
		double[][] training_features = datasets[0]; // transformed training features
		double[][] testing_features = datasets[1]; // transformed testing features
		int[] training_labels = extract_labels(read_csv(FILE_PATH_1));
		int[] testing_labels = extract_labels(read_csv(FILE_PATH_2));

		// initialise variables to track the best accuracy and hyperparameter
		// combination
		double best_accuracy = 0;
		double[] best_params = new double[6]; // Store best values for LR, Epoch, RS, VP, RF, MT

		// iterate over all combinations of hyperparameters
		for (double learning_rate : LEARNING_RATES) {
			for (int epoch : EPOCHS) {
				for (double regularisation_strength : REGULARISATION_STRENGTHS) {
					for (double violation_penalty : VIOLATION_PENALTIES) {
						for (double regularisation_factor : REGULARISATION_FACTORS) {
							for (double margin_threshold : MARGIN_THRESHOLDS) {
								System.out.printf(
										"Training with LR = %.7f, Epoch = %d, RS = %.2f, VP = %.2f, RF = %.4f, MT = %.2f%n",
										learning_rate, epoch, regularisation_strength, violation_penalty,
										regularisation_factor, margin_threshold);
								// evaluate the model for all classes with the current combination of
								// hyperparameters
								double[] class_accuracies = evaluate_all_classes(training_features, training_labels,
										testing_features, testing_labels, learning_rate, epoch, regularisation_strength,
										violation_penalty, regularisation_factor, margin_threshold);
								// calculate the overall accuracy for the current combination
								double overall_accuracy = calculate_overall_accuracy(class_accuracies);
								System.out.println("accuracy: " + overall_accuracy);

								// update the best parameters
								best_accuracy = update_best_parameters(best_accuracy, overall_accuracy, learning_rate,
										epoch, regularisation_strength, violation_penalty, regularisation_factor,
										margin_threshold, best_params);
							}
						}
					}
				}
			}
		}

		print_best_parameters(best_params[0], (int) best_params[1], best_params[2], best_params[3], best_params[4],
				best_params[5], best_accuracy);
	}
}
