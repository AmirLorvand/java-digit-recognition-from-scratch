/**
 * This program implements a Multi-Layer Perceptrons (MLP) for digit recognition using 
 * handwritten digit datasets. It includes functionality for training the MLP, 
 * evaluating its performance, and optimising hyperparameters through grid search.
 *
 * Key Features:
 * - Reads datasets from CSV files and processes them by extracting features and labels.
 * - Normalises the dataset for improved training performance.
 * - Defines an MLP structure with one hidden layer and configurable hyperparameters.
 * - Uses ReLU activation for the hidden layer and Softmax for the output layer.
 * - Supports forward and backward propagation for training the MLP.
 * - Implements 2-fold cross-validation for robust model evaluation.
 * - Optimises hyperparameters such as learning rate, number of epochs, and hidden neurons 
 *   using a grid search approach.
 * - Calculates performance metrics like accuracy, variance, and a composite score.
 *
 * How to Use:
 * 1. Prepare two datasets in CSV format, with the last column as the labels.
 * 2. Place the CSV files in the working directory and specify their paths in the `main` method.
 * 3. Adjust hyperparameters (learning rates, epochs, hidden neurons) if necessary.
 * 4. Run the program to train, evaluate, and find the best hyperparameters for the MLP.
 * 5. View the results, including the mean accuracy, variance, and best hyperparameter configuration.
 *
 * Example:
 * Input: "dataSet1.csv", "dataSet2.csv"
 * Output: Performance metrics for each hyperparameter combination and the best configuration.
 *
 * Note:
 * - The datasets must be formatted correctly, with consistent dimensions for features and labels.
 * - The program outputs training progress, fold accuracies, and the results of the hyperparameter grid search.
 *
 * Author: Amir Lorvand
 * Date: 9 Dec 2024
 */

package digit_reco_MLP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class MLP {

	// define hyperparameters
	static final double[] LEARNING_RATES = { 0.001, 0.01, 0.05, 0.1 };
	static final int[] EPOCH_VALUES = { 10, 20, 50, 100 };
	static final int[] HIDDEN_LAYER_PERCEPTRONS = { 8, 16, 32, 64 }; // hidden layer sizes
	static final int OUTPUT_PERCEPTRON_NUMBER = 10;
	static final int RUNS_BY_COMBINATION = 5; // number of runs per hyperparameter combination
	static final double VARIANCE_WEIGHTS = 0.1; // variance penalty factor for scoring (keep it low to focus on the
												// mean)
	static final double LARGEST_PIXEL_VALUE = 16.0; // used for normalisation
	static final String FILE_PATH_1 = "dataSet1.csv";
	static final String FILE_PATH_2 = "dataSet2.csv";
	static Random random = new Random(13);

	// reads a CSV file and converts it into a 2D integer array
	static int[][] read_csv(String file_path) {

		ArrayList<int[]> data_list = new ArrayList<>();

		try (Scanner scanner = new Scanner(new File(file_path))) {
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] values = line.split(",");
				int[] int_values = new int[values.length];

				// convert values from String to integer
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

	// extracts input features (pixels) from the dataset
	static int[][] extract_features(int[][] dataset) {

		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;
		// define an array to store the features
		int[][] features = new int[number_row][number_of_features];

		// copy feature values from dataset
		for (int image_index = 0; image_index < features.length; image_index++) {
			for (int pixel_index = 0; pixel_index < features[0].length; pixel_index++) {
				features[image_index][pixel_index] = dataset[image_index][pixel_index];
			}
		}

		return features;
	}

	// a function to extract labels from a dataset
	static int[] extract_labels(int[][] dataset) {

		// find the size of the dataset
		int number_row = dataset.length;
		int number_of_features = 64;

		// define an array to store labels
		int[] labels = new int[number_row];

		// copy label values from the last column of the dataset
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

	// a function to calculate the output of a layer
	static double[] calculate_layer_output(double[] inputs, double[][] weights, double[] biases) {

		// find the number of perceptrons in the layer
		int num_perceptrons = biases.length;
		double[] output = new double[num_perceptrons];

		// for each perceptron in the current layer compute its output
		for (int perceptron_index = 0; perceptron_index < num_perceptrons; perceptron_index++) {
			double weighted_sum = biases[perceptron_index]; // add bias
			for (int input_index = 0; input_index < inputs.length; input_index++) {
				weighted_sum += inputs[input_index] * weights[perceptron_index][input_index]; // add weighted input
			}
			output[perceptron_index] = weighted_sum; // assign weighted sum to output
		}
		return output;
	}

	// a function to initialise weights with random values between -1 to 1
	static double[][] initialise_weights(int num_row, int num_col) {

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

	// a function to initialise biases with values between -1 and 1
	static double[] initialise_biases(int num_biases) {
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
				error_sum += output_layer_error[output_perceptron_index]
						* output_weights[output_perceptron_index][hidden_perceptron_index];
			}
			// apply ReLu derivative
			hidden_layer_error[hidden_perceptron_index] = hidden_layer_output[hidden_perceptron_index] > 0 ? error_sum
					: 0;
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

	// trains the model using one training example
	static void train(double[] inputs, double[] target_output, double[][] hidden_weights, double[] hidden_biases,
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

	// a function to calculate the accuracy of the model predictions
	static double calculate_accuracy(int[] predicted_labels, int[] actual_labels) {

		int correct_predictions = 0;
		double percentage_multiplier = 100.0;
		// compare predicted labels with actual labels
		for (int label_position = 0; label_position < predicted_labels.length; label_position++) {
			if (predicted_labels[label_position] == actual_labels[label_position]) {
				correct_predictions++;
			}
		}
		return (double) correct_predictions / actual_labels.length * percentage_multiplier;
	}

	// predicts the output for a single input
	static double[] predict(double[] inputs, double[][] hidden_weights, double[] hidden_biases,
			double[][] output_weights, double[] output_biases) {

		// using RelU as activation function for the hidden layer
		double[] hidden_layer_output = relu(calculate_layer_output(inputs, hidden_weights, hidden_biases));

		// using Softmax function as activation for output layer
		return softmax(calculate_layer_output(hidden_layer_output, output_weights, output_biases));
	}

	// a function to calculate the mean
	static double calculate_mean(double[] values) {

		double sum = 0;
		for (double value : values) {
			sum += value;
		}
		return sum / values.length;
	}

	// a function to calculate the variance
	static double calculate_variance(double[] values) {

		double mean = calculate_mean(values);
		double square_difference_sum = 0;
		for (double value : values) {
			square_difference_sum += Math.pow(value - mean, 2);
		}
		return square_difference_sum / values.length;
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

	// evaluates the model on a testing dataset and calculates accuracy
	static double evaluate_model(double[][] testing_features, int[] testing_labels, double[][] hidden_weights,
			double[] hidden_biases, double[][] output_weights, double[] output_biases) {

		int[] predictions = new int[testing_features.length];

		// predict the label for each test sample
		for (int sample_index = 0; sample_index < testing_features.length; sample_index++) {
			double[] output = predict(testing_features[sample_index], hidden_weights, hidden_biases, output_weights,
					output_biases);
			predictions[sample_index] = get_max_index(output); // convert probabilities to label
		}
		return calculate_accuracy(predictions, testing_labels); // return accuracy
	}

	// executes the training process for the specified number of epochs
	static void execute_training(double[][] training_features, int[] training_labels, double[][] hidden_weights,
			double[] hidden_biases, double[][] output_weights, double[] output_biases, double learning_rate,
			int epochs) {

		for (int epoch = 0; epoch < epochs; epoch++) {
			// train on each sample in the training set
			for (int sample_index = 0; sample_index < training_features.length; sample_index++) {

				// one hot encoding for target output
				double[] target_output = new double[OUTPUT_PERCEPTRON_NUMBER];
				target_output[training_labels[sample_index]] = 1; // set the target class to 1
				// train the model on the sample
				train(training_features[sample_index], target_output, hidden_weights, hidden_biases, output_weights,
						output_biases, learning_rate);
			}
		}
	}

	// a function to train the model on one fold and evaluate it on the other fold
	static double train_evaluate_fold(double[][] training_features, int[] training_labels, double[][] testing_features,
			int[] testing_labels, double learning_rate, int epochs, int hidden_perceptrons) {

		// initialise weights and biases
		double[][] hidden_weights = initialise_weights(hidden_perceptrons, training_features[0].length);
		double[] hidden_biases = initialise_biases(hidden_perceptrons);
		double[][] output_weights = initialise_weights(OUTPUT_PERCEPTRON_NUMBER, hidden_perceptrons);
		double[] output_biases = initialise_biases(OUTPUT_PERCEPTRON_NUMBER);

		// train the model
		execute_training(training_features, training_labels, hidden_weights, hidden_biases, output_weights,
				output_biases, learning_rate, epochs);

		// evaluate the model
		return evaluate_model(testing_features, testing_labels, hidden_weights, hidden_biases, output_weights,
				output_biases);
	}

	// performs 2-fold cross-validation and calculates average accuracy
	static double two_fold_validation(double[][] features_1, int[] labels_1, double[][] features_2, int[] labels_2,
			double learning_rate, int epochs, int hidden_perceptrons) {

		int number_of_folds = 2;

		// train and evaluate on the first fold
		double accuracy_1 = train_evaluate_fold(features_1, labels_1, features_2, labels_2, learning_rate, epochs,
				hidden_perceptrons);

		// roles reversed (second fold)
		double accuracy_2 = train_evaluate_fold(features_2, labels_2, features_1, labels_1, learning_rate, epochs,
				hidden_perceptrons);

		return (accuracy_1 + accuracy_2) / number_of_folds;
	}

	// a function to evaluate a combination of hyperparameters (return the score)
	static double[] evaluate_hyperparameters(double[][] training_features, int[] training_labels,
			double[][] testing_features, int[] testing_labels, double learning_rate, int epochs,
			int hidden_perceptrons) {

		// define an array to store accuracies for each run
		double[] accuracies = new double[RUNS_BY_COMBINATION];
		// perform multiple runs for the same hyperparameter combination
		for (int run = 0; run < RUNS_BY_COMBINATION; run++) {
			accuracies[run] = two_fold_validation(training_features, training_labels, testing_features, testing_labels,
					learning_rate, epochs, hidden_perceptrons);
		}

		// calculate mean and variance
		double mean_accuracy = calculate_mean(accuracies);
		double variance_accuracy = calculate_variance(accuracies);

		// calculate the score
		double score = mean_accuracy - (VARIANCE_WEIGHTS * variance_accuracy);

		return new double[] { mean_accuracy, variance_accuracy, score };
	}

	// a function to print each combination of hyperparameters result
	static void print_hyperparameter_result(double learning_rate, int epochs, int hidden_perceptrons,
			double[] results) {

		System.out.printf(
				"LR: %.3f, epochs: %d, hidden neurons: %d -> mean ACCURACY: %.2f%%, variance: %.2f, score: %.2f%n",
				learning_rate, epochs, hidden_perceptrons, results[0], results[1], results[2]);
	}

	// a function to print the final result
	static void print_best_hyperparameters(double[] best_parameters, double best_mean_accuracy, double best_variance,
			double best_score) {

		System.out.println("---------------------------------------------");
		System.out.println("<< best hyperparameters combination >>");
		System.out.printf("learning rate: %.5f%n", best_parameters[0]);
		System.out.printf("number of epochs: %d%n", (int) best_parameters[1]);
		System.out.printf("number of perceptrons in hidden layer: %d%n", (int) best_parameters[2]);
		System.out.printf("with average accuracy: %.3f%%%n", best_mean_accuracy);
		System.out.printf("variance: %.3f%n", best_variance);
		System.out.printf("score: %.3f%n", best_score);
	}

	// a function to run the grid search
	static void evaluate_hyperparameters_grid(double[][] training_features, int[] training_labels,
			double[][] testing_features, int[] testing_labels) {

		// define an array to store best hyperparameters {learning_rate, epochs,
		// hidden_perceptrons}
		double[] best_parameters = new double[3];
		double best_score = Double.NEGATIVE_INFINITY, best_mean_accuracy = 0, best_variance = 0;

		for (double learning_rate : LEARNING_RATES) {
			for (int epochs : EPOCH_VALUES) {
				for (int hidden_perceptrons : HIDDEN_LAYER_PERCEPTRONS) {

					// evaluate each combination
					double[] results = evaluate_hyperparameters(training_features, training_labels, testing_features,
							testing_labels, learning_rate, epochs, hidden_perceptrons);

					// print result for this combination
					print_hyperparameter_result(learning_rate, epochs, hidden_perceptrons, results);

					// track the best results
					if (results[2] > best_score) {
						best_score = results[2];
						best_parameters = new double[] { learning_rate, epochs, hidden_perceptrons };
						best_mean_accuracy = results[0];
						best_variance = results[1];
					}
				}
			}
		}
		print_best_hyperparameters(best_parameters, best_mean_accuracy, best_variance, best_score);
	}

	public static void main(String[] args) {

		// extract features and labels, and normalise them
		double[][] features_1 = normalise(extract_features(read_csv(FILE_PATH_1)));
		double[][] features_2 = normalise(extract_features(read_csv(FILE_PATH_2)));
		int[] labels_1 = extract_labels(read_csv(FILE_PATH_1));
		int[] labels_2 = extract_labels(read_csv(FILE_PATH_2));

		// run the grid search
		evaluate_hyperparameters_grid(features_1, labels_1, features_2, labels_2);

	}
}
