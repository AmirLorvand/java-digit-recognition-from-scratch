/**
 * This program implements a Support Vector Machine (SVM) for multi-class digit recognition
 * using handwritten digit datasets. The SVM employs the Radial Basis Function (RBF) kernel
 * and a one-vs-all strategy for classification. It includes functionality for training, 
 * evaluating, and optimising hyperparameters through grid search.
 *
 * Key Features:
 * - Reads datasets from CSV files and processes them by extracting features and labels.
 * - Normalises the dataset for improved model performance.
 * - Implements RBF kernel computation and dual SVM training using gradient ascent.
 * - Uses a one-vs-all strategy for multi-class classification (10 classes: 0-9).
 * - Supports 2-fold cross-validation for robust evaluation.
 * - Provides grid search functionality to optimise hyperparameters:
 *   * Kernel width
 *   * Regularisation parameter
 *   * Learning rate
 *   * Number of epochs
 * - Tracks and outputs the best hyperparameter combination based on accuracy.
 *
 * Program Execution Notes:
 * - Running the full grid search with all hyperparameter combinations is computationally
 *   expensive and may take a significant amount of time (up to 24 hours).
 * - To reduce execution time, use the following hyperparameter combination for optimal results:
 *   * KERNEL_WIDTHS = {0.38}
 *   * REGULARISATION_PARAMS = {250}
 *   * LEARNING_RATES = {0.0012}
 *   * NUMBER_EPOCHS = {1000}
 * - Adjust these hyperparameters in the `main` method as necessary.
 *
 * How to Use:
 * 1. Prepare two datasets in CSV format, with the last column as the labels.
 * 2. Place the CSV files in the working directory and specify their paths in the `main` method.
 * 3. Optionally adjust hyperparameter arrays for grid search.
 * 4. Run the program to train, evaluate, and find the best hyperparameters for the SVM.
 * 5. View the results, including the best hyperparameter combination and corresponding accuracy.
 *
 * Example:
 * Input: "dataSet1.csv", "dataSet2.csv"
 * Output: Grid search progress, fold accuracies, and best hyperparameter configuration.
 *
 * Note:
 * - Ensure that datasets are formatted consistently with features and labels.
 * - The program prints intermediate results for each hyperparameter combination.
 *
 * Author: Amir Lorvand
 * Date: 9 Dec 2024
 */

package digit_reco_SVM;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class SVM {

	// define hyperparameters
	static final double[] KERNEL_WIDTHS = { 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.32, 0.35, 0.37, 0.38, 0.39, 0.4 };
	static final double[] REGULARISATION_PARAMS = { 1, 10, 100, 125, 150, 175, 200, 225, 250, 275 };
	static final double[] LEARNING_RATES = { 0.0005, 0.0008, 0.001, 0.0012, 0.0013, 0.0015 };
	static final int[] NUMBER_EPOCHS = { 100, 250, 500, 1000 };
	static final double LARGEST_PIXEL_VALUE = 16.0; // max pixel value for normalisation
	static final double PERCENTAGE_MULTIPLIER = 100.0; // multiplier to convert to percentage
	static final int TRUE_CLASS = 1; // value for true class in one-vs-all labels
	static final int FALSE_CLASS = -1; // value for other classes in one-vs-all labels
	static final String FILE_PATH_1 = "dataSet1.csv";
	static final String FILE_PATH_2 = "dataSet2.csv";
	static final int NUMBER_OF_CLASSES = 10;

	// reads a dataset from a CSV file and converts it into a 2D integer array
	static int[][] read_csv(String file_path) {

		ArrayList<int[]> data_list = new ArrayList<>();

		try (Scanner scanner = new Scanner(new File(file_path))) {

			while (scanner.hasNextLine()) {

				String line = scanner.nextLine();
				String[] values = line.split(",");
				int[] int_values = new int[values.length];
				// convert each value to integer
				for (int column_index = 0; column_index < values.length; column_index++) {
					int_values[column_index] = Integer.parseInt(values[column_index]);
				}
				data_list.add(int_values);
			}
		} catch (FileNotFoundException exception) {
			exception.printStackTrace();
		}

		// convert array list to an array
		int[][] data_array = new int[data_list.size()][];

		for (int row_index = 0; row_index < data_list.size(); row_index++) {
			data_array[row_index] = data_list.get(row_index);
		}

		return data_array;
	}

	// extracts input features (pixel values) from the dataset
	static int[][] extract_features(int[][] dataset) {

		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;
		// define an array to store the features
		int[][] features = new int[number_row][number_of_features];

		// extract the first 64 values of each row as features
		for (int image_index = 0; image_index < features.length; image_index++) {
			for (int pixel_index = 0; pixel_index < features[0].length; pixel_index++) {
				features[image_index][pixel_index] = dataset[image_index][pixel_index];
			}
		}
		return features;
	}

	// extracts the labels (last column) from the dataset
	static int[] extract_labels(int[][] dataset) {

		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;

		// define an array to store labels
		int[] labels = new int[number_row];
		// extract the last value of each row as the label
		for (int image_index = 0; image_index < labels.length; image_index++) {
			labels[image_index] = dataset[image_index][number_of_features];
		}
		return labels;
	}

	// a function to normalise dataset (divide by the largest value = 16)
	static double[][] normalise(int[][] dataset) {

		double[][] normalised_dataset = new double[dataset.length][dataset[0].length];
		// normalise each pixel value
		for (int image_index = 0; image_index < dataset.length; image_index++) {
			for (int pixel_index = 0; pixel_index < dataset[0].length; pixel_index++) {
				normalised_dataset[image_index][pixel_index] = dataset[image_index][pixel_index] / LARGEST_PIXEL_VALUE;
			}
		}
		return normalised_dataset;
	}

	// computes the Radial Basis Function (RBF) kernel value between two vectors
	static double rbf(double[] vector_1, double[] vector_2, double kernel_width) {

		double sum = 0;
		// compute the squared euclidean distance
		for (int dimension_index = 0; dimension_index < vector_1.length; dimension_index++) {
			sum += Math.pow(vector_1[dimension_index] - vector_2[dimension_index], 2);
		}
		// apply the RBF formula
		return Math.exp(sum * (-kernel_width));
	}

	// calculates the RBF kernel matrix for a given set of training features
	static double[][] calculate_rbf_matrix(double[][] training_features, double kernel_width) {

		double[][] rbf_matrix = new double[training_features.length][training_features.length];
		// compute RBF values for all pairs of training samples
		for (int row_index = 0; row_index < training_features.length; row_index++) {
			for (int column_index = 0; column_index < training_features.length; column_index++) {
				rbf_matrix[row_index][column_index] = rbf(training_features[row_index], training_features[column_index],
						kernel_width);
			}
		}
		return rbf_matrix;
	}

	// trains the SVM using the dual form of optimisation (adjusts alpha values)
	static void train_svm_dual(double[][] training_features, int[] training_labels, double[] dual_coefficients,
			double[] bias, double[][] rbf_matrix, double regularisation_param, double learning_rate, int epochs) {

		for (int epoch = 0; epoch < epochs; epoch++) {
			// loop over training samples
			for (int sample_index = 0; sample_index < training_features.length; sample_index++) {

				// calculate decision value for current sample
				double decision = 0;
				for (int neighbor_index = 0; neighbor_index < training_features.length; neighbor_index++) {
					decision += training_labels[neighbor_index] * rbf_matrix[sample_index][neighbor_index]
							* dual_coefficients[neighbor_index];
				}
				decision += bias[0];

				// update dual coefficients (alpha) and bias on margin
				if (decision * training_labels[sample_index] < 1) {
					dual_coefficients[sample_index] += (regularisation_param - dual_coefficients[sample_index])
							* learning_rate;
					bias[0] += training_labels[sample_index] * learning_rate;
				}
			}
		}
	}

	// calculates the decision scores for each class using the RBF kernel
	static double[] calculate_decision_scores(double[][] training_features, int[] training_labels,
			double[] testing_features, double[][] dual_coefficients, double[] bias, double kernel_width) {

		double[] decision_scores = new double[dual_coefficients.length];

		// calculate each class decision score
		for (int class_index = 0; class_index < dual_coefficients.length; class_index++) {
			decision_scores[class_index] = bias[class_index];

			// add contributions from all training samples
			for (int training_sample_index = 0; training_sample_index < training_features.length; training_sample_index++) {
				decision_scores[class_index] += ((training_labels[training_sample_index] == class_index) ? TRUE_CLASS
						: FALSE_CLASS) * rbf(training_features[training_sample_index], testing_features, kernel_width)
						* dual_coefficients[class_index][training_sample_index];
			}
		}
		return decision_scores;
	}

	// a function to find the class with the highest decision score
	static int get_predicted_class(double[] decision_scores) {

		int prediction = 0;
		double max_score = decision_scores[0];
		// iterate through scores to find the class with the highest score
		for (int class_index = 0; class_index < decision_scores.length; class_index++) {

			if (decision_scores[class_index] > max_score) {
				prediction = class_index; // update prediction
				max_score = decision_scores[class_index];
			}
		}
		return prediction;
	}

	// predicts the class for a single test sample
	static int predict(double[][] training_features, int[] training_labels, double[] testing_features,
			double[][] dual_coefficients, double[] bias, double kernel_width) {

		// calculate decision score for all classes
		double[] decision_scores = calculate_decision_scores(training_features, training_labels, testing_features,
				dual_coefficients, bias, kernel_width);

		// return the predicted class with the highest score
		return get_predicted_class(decision_scores);
	}

	// generates binary labels for one-vs-all classification for a specific class
	static int[] generate_one_vs_all_labels(int[] labels, int target_class) {

		int[] binary_labels = new int[labels.length];

		// assign 1 for target class and -1 for other classes
		for (int label_index = 0; label_index < labels.length; label_index++) {
			binary_labels[label_index] = (labels[label_index] == target_class) ? TRUE_CLASS : FALSE_CLASS;
		}
		return binary_labels;
	}

	// trains the SVM using the one-vs-all strategy
	static void train_one_vs_all(double[][] training_features, int[] training_labels, double[][] dual_coefficients,
			double[] bias, double kernel_width, double regularisation_param, double learning_rate, int epochs) {

		// iterate over 0 to 9 (all the classes) to train each of them
		for (int class_label = 0; class_label < NUMBER_OF_CLASSES; class_label++) {

			// create binary class for the selected class
			int[] binary_labels = generate_one_vs_all_labels(training_labels, class_label);
			double[][] kernel_matrix = calculate_rbf_matrix(training_features, kernel_width); // compute the RBF kernel
																								// matrix for the
																								// training data

			// train the SVM for the selected class
			train_svm_dual(training_features, binary_labels, dual_coefficients[class_label],
					new double[] { bias[class_label] }, kernel_matrix, regularisation_param, learning_rate, epochs);
		}
	}

	// evaluates the model accuracy on a testing dataset
	static double evaluate_model_accuracy(double[][] trainig_features, int[] training_labels,
			double[][] testing_features, int[] testing_labels, double[][] dual_coefficients, double[] bias,
			double kernel_width) {

		int correct_predictions = 0;
		// predict the class for each test sample
		for (int sample_index = 0; sample_index < testing_features.length; sample_index++) {

			int prediction = predict(trainig_features, training_labels, testing_features[sample_index],
					dual_coefficients, bias, kernel_width);
			// increase the correct prediction counter if the prediction matches the true
			// label
			if (prediction == testing_labels[sample_index]) {
				correct_predictions++;
			}
		}
		return (double) correct_predictions / testing_features.length * PERCENTAGE_MULTIPLIER; // return accuracy as a
																								// percentage
	}

	// trains and evaluates the SVM on a single fold of data
	static double evaluate_fold_accuracy(double[][] training_features, int[] training_labels,
			double[][] testing_features, int[] testing_labels, double kernel_width, double regularisation_param,
			double learning_rate, int epochs) {

		// initialise dual coefficients and biases
		double[][] dual_coefficients = new double[10][training_features.length];
		double[] bias = new double[10];

		// train the model using one-vs-all strategy
		train_one_vs_all(training_features, training_labels, dual_coefficients, bias, kernel_width,
				regularisation_param, learning_rate, epochs);

		// evaluate the model on testing data and return accuracy
		return evaluate_model_accuracy(training_features, training_labels, testing_features, testing_labels,
				dual_coefficients, bias, kernel_width);
	}

	// Performs 2-fold cross-validation and returns the average accuracy
	static double cross_validate_hyperparameters(double[][] features_1, int[] labels_1, double[][] features_2,
			int[] labels_2, double kernel_width, double regularisation_param, double learning_rate, int epochs) {

		System.out.printf("testing with kernel_width=%.3f, regularisation_param=%.2f, learning_rate=%.4f, epochs=%d%n",
				kernel_width, regularisation_param, learning_rate, epochs);

		double number_of_folds = 2.0;

		// evaluate first fold
		double accuracy_1 = evaluate_fold_accuracy(features_1, labels_1, features_2, labels_2, kernel_width,
				regularisation_param, learning_rate, epochs);

		// roles reversed (second fold)
		double accuracy_2 = evaluate_fold_accuracy(features_2, labels_2, features_1, labels_1, kernel_width,
				regularisation_param, learning_rate, epochs);

		System.out.printf("Accuracy: %.2f%%%n", (accuracy_1 + accuracy_2) / number_of_folds);

		return (accuracy_1 + accuracy_2) / number_of_folds;
	}

	// a function to update the best hyperparameters and track accuracy
	static void update_best_hyperparameters(double accuracy, double kernel_width, double regularisation_param,
			double learning_rate, int epochs, double[] best_hyperparameters) {

		if (accuracy > best_hyperparameters[0]) {
			best_hyperparameters[0] = accuracy;
			best_hyperparameters[1] = kernel_width;
			best_hyperparameters[2] = regularisation_param;
			best_hyperparameters[3] = learning_rate;
			best_hyperparameters[4] = epochs;
		}
	}

	// performs grid search to find the best combination of hyperparameters
	static void grid_search(double[][] features_1, int[] labels_1, double[][] features_2, int[] labels_2) {

		// track {accuracy, kernel_width, regularisation_param, learning_rate, epochs}
		double[] best_hyperparameters = { 0, 0, 0, 0, 0 };

		for (double kernel_width : KERNEL_WIDTHS) {
			for (double regularisation_param : REGULARISATION_PARAMS) {
				for (double learning_rate : LEARNING_RATES) {
					for (int epochs : NUMBER_EPOCHS) {

						// evaluate the current combination
						double accuracy = cross_validate_hyperparameters(features_1, labels_1, features_2, labels_2,
								kernel_width, regularisation_param, learning_rate, epochs);

						// update the best hyperparameters if it is necessary
						update_best_hyperparameters(accuracy, kernel_width, regularisation_param, learning_rate, epochs,
								best_hyperparameters);
					}
				}
			}
		}
		System.out.printf(
				"best combination: kernel_width=%.4f, regularisation_param=%.2f, learning_rate=%.5f, epochs=%d, ACCURACY=%.3f%%%n",
				best_hyperparameters[1], best_hyperparameters[2], best_hyperparameters[3],
				(int) best_hyperparameters[4], best_hyperparameters[0]);
	}

	public static void main(String[] args) {

		// extract features, labels and normalise features
		double[][] features_1 = normalise(extract_features(read_csv(FILE_PATH_1)));
		double[][] features_2 = normalise(extract_features(read_csv(FILE_PATH_2)));
		int[] labels_1 = extract_labels(read_csv(FILE_PATH_1));
		int[] labels_2 = extract_labels(read_csv(FILE_PATH_2));

		// perform grid search to find the best hyperparameters
		grid_search(features_1, labels_1, features_2, labels_2);
	}
}
