/**
 * This program implements a Voting System for digit recognition by combining predictions 
 * from three different machine learning models: Nearest Neighbor (NN), Multi-Layer 
 * Perceptron (MLP), and Support Vector Machine (SVM). It uses a voting mechanism to 
 * determine the final predicted label for each sample, aiming to enhance overall accuracy.
 *
 * Key Features:
 * - Reads datasets from CSV files and processes them by extracting features and labels.
 * - Implements and trains individual models:
 *   * Nearest Neighbour (NN): Based on Euclidean distance.
 *   * Multi-Layer Perceptron (MLP): Single hidden layer with configurable perceptrons.
 *   * Support Vector Machine (SVM): RBF kernel with dual optimisation for multi-class classification.
 * - Normaliswes the dataset for MLP and SVM models.
 * - Combines predictions from NN, MLP, and SVM using a majority voting system.
 * - Evaluates model accuracy using 2-fold cross-validation.
 * - Tracks and outputs overall accuracy for the voting system.
 *
 * How the Voting System Works:
 * 1. Each model predicts the label for all samples in the testing dataset.
 * 2. The voting mechanism selects the final label:
 *    * If two models agree, their common label is selected.
 *    * If all three models disagree, NN's prediction is selected as a fallback.
 *
 * Program Execution Notes:
 * - Hyperparameters for MLP and SVM models are predefined but can be adjusted as needed:
 *   * MLP: Learning rate, number of epochs, and hidden layer perceptrons.
 *   * SVM: Kernel width, regularisation parameter, learning rate, and number of epochs.
 * - Ensure datasets are formatted consistently with features and labels, and placed in the working directory.
 * - The program outputs fold accuracies and the final overall accuracy after voting.
 *
 * How to Use:
 * 1. Prepare two datasets in CSV format with the last column as the labels.
 * 2. Place the CSV files in the working directory and specify their paths in the `main` method.
 * 3. Run the program to train the models, perform voting, and evaluate accuracy.
 *
 * Example:
 * Input: "dataSet1.csv", "dataSet2.csv"
 * Output: Accuracy for each fold and the overall voting system accuracy.
 *
 * Note:
 * - The program prints intermediate results for training and voting.
 * - Adjust hyperparameters in the `main` method to experiment with different configurations.
 *
 * Author: Amir Lorvand
 * Date: 9 Dec 2024
 */

package voting_system;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class voting_system {

	// define hyperparameters for MLP and SVM models
	static final double LEARNING_RATE_MLP = 0.05;
	static final int EPOCH_MLP = 100;
	static final int HIDDEN_LAYER_PRCEPTRONS = 64;
	static final int OUTPUT_PERCEPTRON_NUMBER = 10;
	
	static final double KERNEL_WIDTH = 0.38;
	static final double REGULARISATION_PARAMS = 250;
	static final double LEARNING_RATES_SVM = 0.0012;
	static final int EPOCH_SVM = 50;
	
	static Random random = new Random(13);
	static final String FILE_PATH_1 = "dataSet1.csv";
	static final String FILE_PATH_2 = "dataSet2.csv";
	static final int TRUE_CLASS = 1; // value for true class in one-vs-all labels
	static final int FALSE_CLASS = -1; // value for other classes in one-vs-all labels
	static final double LARGEST_PIXEL_VALUE = 16.0; // max pixel value for normalisation

	// NEAREST NEIGHBOUR
	// reads a dataset from a CSV file and converts it into a 2D integer array
	static int[][] read_csv(String file_path) {
		ArrayList<int[]> data_list = new ArrayList<>();
		try (Scanner scanner = new Scanner(new File(file_path))) {
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] values = line.split(",");
				int[] int_values = new int[values.length];
				for (int column_index = 0; column_index < values.length; column_index++) {
					int_values[column_index] = Integer.parseInt(values[column_index]); // convert string values to
																						// integers
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

	// a function to calculate Euclidean distance between two points
	static double calculate_distance(int[] point_1, int[] point_2) {
		double squared_difference_sum = 0;
		for (int coordinate_index = 0; coordinate_index < point_1.length; coordinate_index++) {
			squared_difference_sum += Math.pow(point_1[coordinate_index] - point_2[coordinate_index], 2);
		}
		return Math.sqrt(squared_difference_sum);
	}

	// predicts the label of a single test sample using the nearest neighbour
	// approach
	static int predict_NN(int[][] training_features, int[] test_sample, int[] training_labels) {

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

	// a function to predict all samples in the testing dataset
	static int[] predict_all_NN(int[][] training_features, int[][] testing_features, int[] training_labels) {
		
		int[] predictions = new int[testing_features.length];
		// predict the label for each test sample
		for (int test_sample_index = 0; test_sample_index < testing_features.length; test_sample_index++) {
			predictions[test_sample_index] = predict_NN(training_features, testing_features[test_sample_index],
					training_labels);
		}
		return predictions;
	}

	// MLP
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

	// a function to calculate the output of a layer
	static double[] calculate_layer_output(double[] inputs, double[][] weights, double[] biases) {

		// find the number of perceptrons in the layer
		int num_perceptrons = biases.length;
		double[] output = new double[num_perceptrons];

		// for each preceptron in the current layer compute its output
		for (int perceptron_index = 0; perceptron_index < num_perceptrons; perceptron_index++) {
			double weighted_sum = biases[perceptron_index]; // add bias
			for (int input_index = 0; input_index < inputs.length; input_index++) {
				weighted_sum += inputs[input_index] * weights[perceptron_index][input_index]; // add weighted input
			}
			output[perceptron_index] = weighted_sum; // assign weighted sum to output
		}
		return output;
	}

	// a function to initialise weights with random values between -1 to 1 (for MLP model)
	static double[][] initialise_weights_MLP(int num_row, int num_col) {
		// define an array to store weights
		double[][] weights = new double[num_row][num_col];

		// assign random values to each weight
		for (int row_index = 0; row_index < weights.length; row_index++) {
			for (int col_index = 0; col_index < weights[0].length; col_index++) {
				weights[row_index][col_index] = random.nextDouble() * 2 - 1; // random value between -1 and 1
			}
		}
		return weights;
	}

	// a function to initialise biases with values between -1 and 1 (for MLP model)
	static double[] initialise_biases_MLP(int num_biases) {
		// array to store biases
		double[] biases = new double[num_biases];

		// assign random value to each bias
		for (int bias_index = 0; bias_index < num_biases; bias_index++) {
			biases[bias_index] = random.nextDouble() * 2 - 1; // random value between -1 and 1
		}
		return biases;
	}

	// ReLu function
	static double[] relu(double[] inputs) {

		for (int input_index = 0; input_index < inputs.length; input_index++) {
			inputs[input_index] = Math.max(0, inputs[input_index]); // replace negative values with 0
		}
		return inputs;
	}

	// Softmax function
	static double[] softmax(double[] inputs) {

		double sum = 0;
		// compute the sum of exponentials
		for (double input : inputs) {
			sum += Math.exp(input);
		}
		// normalise each input value
		for (int input_index = 0; input_index < inputs.length; input_index++) {
			inputs[input_index] = Math.exp(inputs[input_index]) / sum;
		}
		return inputs;
	}

	// performs a forward pass for the hidden layer using ReLU activation
	static double[] forward_hidden_layer(double[] inputs, double[][] hidden_weights, double[] hidden_biases) {

		return relu(calculate_layer_output(inputs, hidden_weights, hidden_biases));
	}

	// performs a forward pass for the output layer using Softmax activation
	static double[] forward_output_layer(double[] hidden_layer_output, double[][] output_weights,
			double[] output_biases) {
		return softmax(calculate_layer_output(hidden_layer_output, output_weights, output_biases));
	}

	// a function to calculate output layer error
	static double[] calculate_output_error(double[] predicted_output, double[] target_output) {

		double[] output_layer_error = new double[target_output.length];
		// compute error for each perceptron in the output layer
		for (int output_perceptron_index = 0; output_perceptron_index < target_output.length; output_perceptron_index++) {
			output_layer_error[output_perceptron_index] = predicted_output[output_perceptron_index]
					- target_output[output_perceptron_index];
		}
		return output_layer_error;
	}

	// a function to calculate hidden layer error
	static double[] calculate_hidden_error(double[] hidden_layer_output, double[] output_layer_error,
			double[][] output_weights) {

		double[] hidden_layer_error = new double[hidden_layer_output.length];
		// compute error for each perceptron in the hidden layer
		for (int hidden_perceptron_index = 0; hidden_perceptron_index < hidden_layer_output.length; hidden_perceptron_index++) {
			double error_sum = 0;
			for (int output_perceptron_index = 0; output_perceptron_index < output_layer_error.length; output_perceptron_index++) {
				error_sum += output_layer_error[output_perceptron_index] * output_weights[output_perceptron_index][hidden_perceptron_index];
			}
			// apply ReLu derivative
			hidden_layer_error[hidden_perceptron_index] = hidden_layer_output[hidden_perceptron_index] > 0 ? error_sum : 0;
		}
		return hidden_layer_error;
	}

	// updates the weights and biases of the output layer using the gradients
	static void update_output_layer_weights(double[][] output_weights, double[] output_biases,
			double[] hidden_layer_output, double[] output_layer_error, double learning_rate) {

		// update weights for each perceptron in the output layer
		for (int output_perceptron_index = 0; output_perceptron_index < output_weights.length; output_perceptron_index++) {
			for (int hidden_perceptron_index = 0; hidden_perceptron_index < hidden_layer_output.length; hidden_perceptron_index++) {
				// gradient descent step for weights
				output_weights[output_perceptron_index][hidden_perceptron_index] -= learning_rate
						* output_layer_error[output_perceptron_index] * hidden_layer_output[hidden_perceptron_index];
			}
			// gradient descent step for biases
			output_biases[output_perceptron_index] -= learning_rate * output_layer_error[output_perceptron_index];
		}
	}

	// updates the weights and biases of the hidden layer using the gradients
	static void update_hidden_layer_weights(double[][] hidden_weights, double[] hidden_biases, double[] inputs,
			double[] hidden_layer_error, double learning_rate) {
		// update weights for each perceptron in the hidden layer
		for (int hidden_perceptron_index = 0; hidden_perceptron_index < hidden_weights.length; hidden_perceptron_index++) {
			for (int input_index = 0; input_index < inputs.length; input_index++) {
				// gradient descent step for weights
				hidden_weights[hidden_perceptron_index][input_index] -= learning_rate
						* hidden_layer_error[hidden_perceptron_index] * inputs[input_index];
			}
			// gradient descent step for biases
			hidden_biases[hidden_perceptron_index] -= learning_rate * hidden_layer_error[hidden_perceptron_index];
		}
	}

	// a function to train the MLP model
	static void train_MLP(double[] inputs, double[] target_output, double[][] hidden_weights, double[] hidden_biases,
			double[][] output_weights, double[] output_biases, double learning_rate) {

		// forward pass
		double[] hidden_layer_output = forward_hidden_layer(inputs, hidden_weights, hidden_biases);
		double[] predicted_output = forward_output_layer(hidden_layer_output, output_weights, output_biases);

		// calculate the error gradients
		double[] output_layer_error = calculate_output_error(predicted_output, target_output);
		double[] hidden_layer_error = calculate_hidden_error(hidden_layer_output, output_layer_error, output_weights);

		// update weights and biases
		update_output_layer_weights(output_weights, output_biases, hidden_layer_output, output_layer_error,
				learning_rate);
		update_hidden_layer_weights(hidden_weights, hidden_biases, inputs, hidden_layer_error, learning_rate);
	}

	// a function to execute training process for an MLP
	static void execute_training_MLP(double[][] training_features, int[] training_labels, double[][] hidden_weights,
			double[] hidden_biases, double[][] output_weights, double[] output_biases, double learning_rate,
			int epochs) {

		for (int epoch = 0; epoch < epochs; epoch++) {
			// iterate through each sample in the training dataset
			for (int sample_index = 0; sample_index < training_features.length; sample_index++) {

				// one hot encoding for target output
				double[] target_output = new double[OUTPUT_PERCEPTRON_NUMBER];
				target_output[training_labels[sample_index]] = 1;

				// train the MLP with the current training sample and its corresponding target output
				train_MLP(training_features[sample_index], target_output, hidden_weights, hidden_biases, output_weights,
						output_biases, learning_rate);
			}
		}
	}

	// a function to find the largest value index in an array
	static int get_max_index(double[] array) {

		int max_index = 0;
		for (int current_index = 1; current_index < array.length; current_index++) {
			if (array[current_index] > array[max_index]) {
				max_index = current_index;
			}
		}
		return max_index;
	}

	// a function to predict the output using a trained MLP
	static double[] predict_MLP(double[] inputs, double[][] hidden_weights, double[] hidden_biases,
			double[][] output_weights, double[] output_biases) {

		// using RelU as activation function calculate the output for the hidden layer
		double[] hidden_layer_output = relu(calculate_layer_output(inputs, hidden_weights, hidden_biases));

		// using Softmax function as activation calculate the output for final layer
		return softmax(calculate_layer_output(hidden_layer_output, output_weights, output_biases));
	}

	// a function to return the predictions from the MLP model for a set of testing features
	static int[] evaluate_model_MLP(double[][] testing_features, double[][] hidden_weights, double[] hidden_biases,
			double[][] output_weights, double[] output_biases) {

		// array to store the predicted class labels for each testing sample
		int[] predictions = new int[testing_features.length];

		// iterate through each sample in the testing dataset
		for (int sample_index = 0; sample_index < testing_features.length; sample_index++) {
			// get the output probabilities for the current sample using the MLP
			double[] output = predict_MLP(testing_features[sample_index], hidden_weights, hidden_biases, output_weights, output_biases);

			// determine the predicted class by finding the index of the maximum output value
			predictions[sample_index] = get_max_index(output);
		}
		return predictions;
	}

	// SVM
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
				rbf_matrix[row_index][column_index] = rbf(training_features[row_index], training_features[column_index], kernel_width);
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
					decision += training_labels[neighbor_index] * rbf_matrix[sample_index][neighbor_index] * dual_coefficients[neighbor_index];
				}
				decision += bias[0];

				// update dual coefficients (alpha) and bias on margin
				if (decision * training_labels[sample_index] < 1) {
					dual_coefficients[sample_index] += (regularisation_param - dual_coefficients[sample_index]) * learning_rate;
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
				decision_scores[class_index] += ((training_labels[training_sample_index] == class_index) ? 1 : -1)
						* rbf(training_features[training_sample_index], testing_features, kernel_width)
						* dual_coefficients[class_index][training_sample_index];
			}
		}
		return decision_scores;
	}

	// a function to find the predicted class with the highest decision score
	static int get_predicted_class_SVM(double[] decision_scores) {

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

	// predicts the class for a single test sample (for SVM)
	static int predict_SVM(double[][] training_features, int[] training_labels, double[] testing_features,
			double[][] dual_coefficients, double[] bias, double kernel_width) {

		// calculate decision score for all classes
		double[] decision_scores = calculate_decision_scores(training_features, training_labels, testing_features,
				dual_coefficients, bias, kernel_width);

		// return the predicted class with the highest score
		return get_predicted_class_SVM(decision_scores);
	}

	// generates binary labels for one-vs-all classification for a specific class
	static int[] generate_one_vs_all_labels(int[] labels, int target_class) {

		int[] binary_labels = new int[labels.length];

		// assign 1 for target class and -1 for others classes
		for (int label_index = 0; label_index < labels.length; label_index++) {
			binary_labels[label_index] = (labels[label_index] == target_class) ? TRUE_CLASS : FALSE_CLASS;
		}
		return binary_labels;
	}

	// trains the SVM using the one-vs-all strategy
	static void train_one_vs_all_SVM(double[][] training_features, int[] training_labels, double[][] dual_coefficients,
			double[] bias, double kernel_width, double regularisation_param, double learning_rate, int epochs) {

		// iterate over 0 to 9 (all the classes) to train each of them
		for (int class_label = 0; class_label < 10; class_label++) {

			// create binary class for the selected class
			int[] binary_labels = generate_one_vs_all_labels(training_labels, class_label);
			double[][] kernel_matrix = calculate_rbf_matrix(training_features, kernel_width);

			// train the SVM for the selected class
			train_svm_dual(training_features, binary_labels, dual_coefficients[class_label],
					new double[] { bias[class_label] }, kernel_matrix, regularisation_param, learning_rate, epochs);
		}
	}

	// evaluates the model accuracy on a testing dataset
	static int[] evaluate_model_SVM(double[][] trainig_features, int[] training_labels, double[][] testing_features,
			double[][] dual_coefficients, double[] bias, double kernel_width) {

		int[] predictions = new int[testing_features.length];
		// predict the class for each test sample
		for (int sample_index = 0; sample_index < testing_features.length; sample_index++) {

			// predict the class of the current sample
			predictions[sample_index] = predict_SVM(trainig_features, training_labels, testing_features[sample_index],
					dual_coefficients, bias, kernel_width);
		}
		return predictions;
	}

	// VOTING SYSTEM
	// a function to train MLP model and return its prediction
	static int[] return_MLP_prediction(double[][] training_features_normalised, int[] training_labels, double[][] testing_features_normalised) {

		// initialise weights and biases
		double[][] hidden_weights = initialise_weights_MLP(HIDDEN_LAYER_PRCEPTRONS, training_features_normalised[0].length);
		double[] hidden_biases = initialise_biases_MLP(HIDDEN_LAYER_PRCEPTRONS);
		double[][] output_weights = initialise_weights_MLP(OUTPUT_PERCEPTRON_NUMBER, HIDDEN_LAYER_PRCEPTRONS);
		double[] output_biases = initialise_biases_MLP(OUTPUT_PERCEPTRON_NUMBER);

		// train the MLP model
		execute_training_MLP(training_features_normalised, training_labels, hidden_weights, hidden_biases,
				output_weights, output_biases, LEARNING_RATE_MLP, EPOCH_MLP);

		// return prediction
		return evaluate_model_MLP(testing_features_normalised, hidden_weights, hidden_biases, output_weights, output_biases);
	}

	// a function to train SVM and return its prediction
	static int[] return_SVM_prediction(double[][] training_features_normalised, int[] training_labels,
			double[][] testing_features_normalised) {

		// initialise dual coefficients and bias
		double[][] dual_coefficients = new double[10][training_features_normalised.length];
		double[] bias = new double[10];

		// train the model
		train_one_vs_all_SVM(training_features_normalised, training_labels, dual_coefficients, bias, KERNEL_WIDTH,
				REGULARISATION_PARAMS, LEARNING_RATES_SVM, EPOCH_SVM);

		// return predictions
		return evaluate_model_SVM(training_features_normalised, training_labels, testing_features_normalised,
				dual_coefficients, bias, KERNEL_WIDTH);
	}

	// a function to perform voting
	static int[] vote(int[] prediction_NN, int[] prediction_MLP, int[] prediction_SVM) {

		// define an array to store votes
		int[] voted_predictions = new int[prediction_NN.length];

		for (int sample_index = 0; sample_index < voted_predictions.length; sample_index++) {

			if (prediction_NN[sample_index] == prediction_MLP[sample_index]) { // NN and MLP agree
				voted_predictions[sample_index] = prediction_NN[sample_index];
			} else if (prediction_NN[sample_index] == prediction_SVM[sample_index]) { // NN and SVM agree
				voted_predictions[sample_index] = prediction_NN[sample_index];
			} else if (prediction_MLP[sample_index] == prediction_SVM[sample_index]) { // MLP and SVM agree
				voted_predictions[sample_index] = prediction_MLP[sample_index];
			} else { // all three disagree, fall back to NN
				voted_predictions[sample_index] = prediction_NN[sample_index];
			}
		}
		return voted_predictions;
	}

	// a function to calculate the accuracy of each fold
	static double calculate_accuracy_fold(int[] voted_predictions, int[] testing_labels) {
		int correct = 0;

		// iterate through each sample in the predictions and testing labels
		for (int sample_index = 0; sample_index < voted_predictions.length; sample_index++) {

			if (voted_predictions[sample_index] == testing_labels[sample_index]) { // check if the predicted label matches the true label
				correct++;
			}
		}
		return (double) correct / testing_labels.length;
	}

	// a function to execute all training and voting
	static double run(int[][] training_features, int[] training_labels, int[][] testing_features,
			int[] testing_labels) {

		// get preditctions from NN, MLP, and SVM
		int[] prediction_NN = predict_all_NN(training_features, testing_features, training_labels);
		int[] prediction_MLP = return_MLP_prediction(normalise(training_features), training_labels,
				normalise(testing_features));
		int[] prediction_SVM = return_SVM_prediction(normalise(training_features), training_labels,
				normalise(testing_features));

		// perform voting using models predictions
		int[] voted_predictions = vote(prediction_NN, prediction_MLP, prediction_SVM);

		// return accuracy
		return calculate_accuracy_fold(voted_predictions, testing_labels);
	}

	public static void main(String[] args) {
		// load datasets
		int[][] dataset_1 = read_csv(FILE_PATH_1);
		int[][] dataset_2 = read_csv(FILE_PATH_2);

		// define a variable to track accuracy
		double total_accuracy = 0;
		for (int fold = 1; fold <= 2; fold++) {
			// split datasets according to the fold
			int[][] training_features = (fold == 1) ? extract_features(dataset_1) : extract_features(dataset_2);
			int[] training_labels = (fold == 1) ? extract_labels(dataset_1) : extract_labels(dataset_2);
			int[][] testing_features = (fold == 1) ? extract_features(dataset_2) : extract_features(dataset_1);
			int[] testing_labels = (fold == 1) ? extract_labels(dataset_2) : extract_labels(dataset_1);

			// accuracy of the fold
			double fold_accuracy = run(training_features, training_labels, testing_features, testing_labels);
			total_accuracy += fold_accuracy;
		}
		System.out.println("Accuracy: " + total_accuracy / 2.0);

	}
}
