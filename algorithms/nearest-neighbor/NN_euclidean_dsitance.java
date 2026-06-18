/**
 * This program implements a Nearest Neighbour classifier using the Euclidean distance 
 * metric to categorise handwritten digit datasets. It supports reading datasets from 
 * CSV files, extracting features and labels, performing predictions, and evaluating 
 * model performance through 2-fold cross-validation.
 *
 * Key Features:
 * - Reads datasets from CSV files and converts them into 2D arrays.
 * - Extracts features and labels from datasets.
 * - Calculates the Euclidean distance between data points.
 * - Performs Nearest Neighbour predictions for single or multiple test samples.
 * - Evaluates model accuracy and supports 2-fold cross-validation.
 * - Outputs the accuracy for each fold and the total cross-validation accuracy.
 *
 * How to Use:
 * 1. Prepare two datasets in CSV format where the last column represents the labels.
 * 2. Place the CSV files in the working directory and provide their paths in the 
 *    `file_path_1` and `file_path_2` variables in the `main` method.
 * 3. Run the program to execute the 2-fold cross-validation and print the results.
 *
 * Note: Ensure that the datasets are formatted correctly with consistent dimensions 
 * for features and labels.
 *
 * Example:
 * Input: "dataSet1.csv", "dataSet2.csv"
 * Output: Fold 1 accuracy, Fold 2 accuracy, and total accuracy.
 *
 * Author: Amir Lorvand
 * Date: 9 Dec 2024
 */

package digit_reco_NN;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class NN_euclidean_dsitance {

	static final String FILE_PATH_1 = "dataSet1.csv";
	static final String FILE_PATH_2 = "dataSet2.csv";

	// reads a dataset from a CSV file and converts it into a 2D integer array
	static int[][] read_csv(String file_path) {

		ArrayList<int[]> data_list = new ArrayList<>();

		try (Scanner scanner = new Scanner(new File(file_path))) {
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] values = line.split(",");
				int[] int_values = new int[values.length];

				for (int column_index = 0; column_index < values.length; column_index++) {
					int_values[column_index] = Integer.parseInt(values[column_index]); // convert string values to integers
				}
				data_list.add(int_values);
			}
		} catch (FileNotFoundException exception) {
			exception.printStackTrace();
		}

		// convert the list to a 2D array
		int[][] data_array = new int[data_list.size()][];

		for (int row_index = 0; row_index < data_list.size(); row_index++) {
			data_array[row_index] = data_list.get(row_index);
		}

		return data_array;
	}

	// extracts only the input features (pixels) from the dataset
	static int[][] extract_features(int[][] dataset) {

		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;
		// define an array to store the features
		int[][] features = new int[number_row][number_of_features];

		// copy the feature values (first 64 columns) from the dataset
		for (int image_index = 0; image_index < features.length; image_index++) {
			for (int pixel_index = 0; pixel_index < features[0].length; pixel_index++) {
				features[image_index][pixel_index] = dataset[image_index][pixel_index];
			}
		}

		return features;
	}

	// extracts the labels from a dataset
	static int[] extract_labels(int[][] dataset) {

		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;

		// define an array to store labels
		int[] labels = new int[number_row];

		// extract the label from the last column (index 64)
		for (int image_index = 0; image_index < labels.length; image_index++) {
			labels[image_index] = dataset[image_index][number_of_features];
		}

		return labels;
	}

	// a function to calculate euclidean distance between two points
	static double calculate_distance(int[] point_1, int[] point_2) {

		double squared_difference_sum = 0;

		for (int coordinate_index = 0; coordinate_index < point_1.length; coordinate_index++) {
			squared_difference_sum += Math.pow(point_1[coordinate_index] - point_2[coordinate_index], 2);
		}

		return Math.sqrt(squared_difference_sum);
	}

	// predicts the label of a single test sample using the nearest neighbour approach
	static int predict(int[][] training_features, int[] test_sample, int[] training_labels) {

		// initialise minimum distance and prediction label
		double min_distance = Double.MAX_VALUE;
		int predicted_label = -1;

		// calculate the distance between the selected image and all training image
		for (int sample_index = 0; sample_index < training_features.length; sample_index++) {

			double distance = calculate_distance(test_sample, training_features[sample_index]);

			// find the corresponding label and update the minimum distance
			if (distance < min_distance) {
				min_distance = distance;
				predicted_label = training_labels[sample_index];
			}
		}
		return predicted_label;
	}

	// a function to predict all the sample in the testing dataset
	static int[] predict_all(int[][] training_features, int[][] testing_features, int[] training_labels) {

		int[] predictions = new int[testing_features.length];

		// predict the label for each test sample
		for (int test_sample_index = 0; test_sample_index < testing_features.length; test_sample_index++) {
			predictions[test_sample_index] = predict(training_features, testing_features[test_sample_index],
					training_labels);
		}

		return predictions;
	}

	// evaluates the model by calculating the accuracy of predictions
	static double evaluate(int[] actual_labels, int[] predicted_labels) {

		int correct_predictions = 0;

		// check if the predicted label is equal to the correct value
		for (int label_index = 0; label_index < actual_labels.length; label_index++) {
			if (predicted_labels[label_index] == actual_labels[label_index]) {
				correct_predictions++;
			}
		}

		double accuracy = (double) correct_predictions / predicted_labels.length;
		return accuracy;
	}

	// evaluates the model for a specific fold and returns the accuracy
	static double evaluate_fold(int[][] features_1, int[] labels_1, int[][] features_2, int[] labels_2, int fold) {

		// determine train and test set based on the fold
		int[][] training_features = (fold == 1) ? features_1 : features_2;
		int[] training_labels = (fold == 1) ? labels_1 : labels_2;
		int[][] testing_features = (fold == 1) ? features_2 : features_1;
		int[] testing_labels = (fold == 1) ? labels_2 : labels_1;

		// predict
		int[] predicted_labels = predict_all(training_features, testing_features, training_labels);

		// evaluation
		return evaluate(testing_labels, predicted_labels);
	}

	// performs 2-fold cross-validation and calculates the total accuracy
	static double cross_validation(int[][] features_1, int[] labels_1, int[][] features_2, int[] labels_2) {

		double total_accuracy = 0;
		int number_of_folds = 2;

		// calculate accuracy in each fold
		for (int fold = 1; fold <= number_of_folds; fold++) {
			double accuracy = evaluate_fold(features_1, labels_1, features_2, labels_2, fold);
			total_accuracy += accuracy;
			System.out.println("fold " + fold + " accuracy: " + accuracy);
		}

		return (double) total_accuracy / number_of_folds;
	}

	// reads datasets and performs cross-validation
	static void execute(String file_path_1, String file_path_2) {

		// load datasets and extract features and labels
		int[][] dataset_1 = read_csv(file_path_1);
		int[][] dataset_2 = read_csv(file_path_2);
		int[][] features_1 = extract_features(dataset_1);
		int[] labels_1 = extract_labels(dataset_1);
		int[][] features_2 = extract_features(dataset_2);
		int[] labels_2 = extract_labels(dataset_2);

		// 2-fold cross validation
		double total_accuracy = cross_validation(features_1, labels_1, features_2, labels_2);

		System.out.println("total accuracy: " + total_accuracy);
	}

	public static void main(String[] args) {

		// execute the program using predefined file paths
		execute(FILE_PATH_1, FILE_PATH_2);
	}
}
