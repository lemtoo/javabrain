package ponder.network;

import ponder.err.PonderInvalidInputException;

import java.util.*;

public class NetworkTrainer
{
  /**
   * A map of feature names to indices within an input set.
   */
  private Map<String, Integer> features = new HashMap<>();

  private List<String> inputFeatures;
  private final String target;

  private double learningRate = 0.3d;

  /**
   * Random number generator for selecting data during training runs.
   */
  private Random rng = new Random();

  /**
   * A 2D array of training data. First index is the training example. Second index is the feature; this index must
   * correspond to the features map.
   */
  private double[][] trainingData = null;

  private final NeuralNetwork neuralNetwork;

  public NetworkTrainer(NeuralNetwork neuralNetwork, List<String> inputFeatures, String target)
  {
    this.neuralNetwork = neuralNetwork;
    this.inputFeatures = inputFeatures;
    this.target = target;
  }

  /**
   * Set the input data. The order of the feature names must match the order of the training data.
   *
   * @param featureNames The names of each feature. Features must be in the same order as the data.
   * @param trainingData The input data for training the network. Each inner array must be the same length, and the
   *                     order of the feature values must match the order of the feature names.
   */
  public void setTrainingData(String[] featureNames, double[][] trainingData) throws PonderInvalidInputException
  {
    int numFeatures = featureNames.length;

    for (double[] trainingDatum : trainingData)
    {
      if(trainingDatum.length != numFeatures)
      {
        throw new PonderInvalidInputException("Training data is not a consistent shape!");
      }
    }

    features.clear();
    for (int featureNum = 0, featureNamesLength = featureNames.length; featureNum < featureNamesLength; featureNum++)
    {
      String featureName = featureNames[featureNum];
      features.put(featureName, featureNum);
    }

    this.trainingData = trainingData;
  }

  /**
   * Run a random data set from the training set list through the network.
   */
  public void runOnce() throws PonderInvalidInputException
  {
    // TODO: Do this reduction before we start doing any training.
    int trainingDataIndex = rng.nextInt(trainingData.length);
    double[] mappedInputFeatures = new double[inputFeatures.size()];

    for (int i = 0; i < inputFeatures.size(); i++)
    {
      mappedInputFeatures[i] = trainingData[trainingDataIndex][features.get(inputFeatures.get(i))];
    }

    // Run the inputs on the network as a whole. There will be one output per node on the final layer.
    final double[] runResults = neuralNetwork.run(mappedInputFeatures);

    double expectedValue = trainingData[trainingDataIndex][features.get(target)];

    // Calculate the error for our target based on all output nodes.
    neuralNetwork.backPropagate(learningRate, new double[]{expectedValue});
  }
}
