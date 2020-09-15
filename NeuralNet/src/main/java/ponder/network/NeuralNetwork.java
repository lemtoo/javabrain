package ponder.network;

import ponder.activation.Activation;
import ponder.err.PonderInvalidInputException;

import java.util.ArrayList;

public class NeuralNetwork
{
  private ArrayList<Layer> layers = new ArrayList<>();

  /**
   *
   * @param numNeurons The number of neurons in this layer.
   * @param numInputs The number of inputs for this layer. This should match either the number of features or the
   *                  number of outputs from the previous layer.
   * @param activatorFunction The activator function to be used for all neurons on this layer.
   * @throws PonderInvalidInputException Indicates invalid inputs.
   */
  public void addLayer(int numNeurons, int numInputs, Activation activatorFunction) throws PonderInvalidInputException
  {
    if(!layers.isEmpty())
    {
      int numPreviousOutputs = layers.get(layers.size() - 1).getNeuronCount();
      if(numInputs != numPreviousOutputs)
      {
        throw new PonderInvalidInputException("Input count for Layer does not match output count for previous layer!");
      }
    }

    layers.add(new Layer(numNeurons, numInputs, activatorFunction, activatorFunction));
  }

  /**
   *
   * @param inputs Run one set of features through the network.
   * @return THe outputs from the run.
   * @throws PonderInvalidInputException Indicates invalid inputs, such as an incorrect input feature count.
   */
  public double[] run(double[] inputs) throws PonderInvalidInputException
  {
    if(inputs.length != layers.get(0).getInputCount())
    {
      throw new PonderInvalidInputException("Input count for run does not match neuron count for the first layer!");
    }

    double[] layerOutputs = inputs;

    for (Layer layer : layers)
    {
      layerOutputs = layer.fireLayer(layerOutputs);
    }

    return layerOutputs;
  }

  /**
   * For the output layer, iterate through the neurons and calculate the error for each neuron as:
   *    (expected - actual) * derivative of activation for the actual
   *
   * For hidden layers, iterate through the neurons (in reverse order) and calculate the error for each neuron as:
   *    The sum of all ouput errors, which are calculated by iterating through the next layer's neurons. Each partial
   *    error is:
   *        the error of the next layer neuron * the weight of the hidden layer neuron WRT that output.
   *      This sum is then multiplied by the derivative of activation for the hidden neuron's actual
   *
   * For each hidden layer, iterate through the neurons and update the weight by adding:
   *    For each input:
   *      Learning rate * the error of the current layer neuron * the input
   * For each hidden layer, iterate through the neurons and update the bias by adding:
   *    Learning rate * neuron's error
   *
   * @param learningRate A small decimal multiplied by the error to prevent oscillation.
   * @param expectedValues The values that should have been produced at the output layer.
   * @throws PonderInvalidInputException
   */
  public void backPropagate(double learningRate, double[] expectedValues) throws PonderInvalidInputException
  {
    // Make sure we have the right number of last-layer outputs.
    final Layer lastLayer = layers.get(layers.size() - 1);
    if(expectedValues.length != lastLayer.getNeuronCount())
    {
      throw new PonderInvalidInputException(
          "The number of expected values does not match the number of output neurons!");
    }

    // --------------------------------------------------
    // Calculate errors
    // --------------------------------------------------

    // Calculate the last layer's error set. For the last layer, this is the difference from the output values.
    // There is no need to calculate partial error because for the output layer, the weights would be all zero, except
    // one weight which would be exactly one. The index of the value 1 corresponds to the index for the output neuron.
    for (int outputIndex = 0; outputIndex < expectedValues.length; outputIndex++)
    {
      final Neuron neuron = lastLayer.getNeurons().get(outputIndex);
      neuron.setLastError(
          (expectedValues[outputIndex] - neuron.getLastOutput())
              * lastLayer.getActivatorFunction().calculateDerivative(neuron.getLastOutput()));
    }

    // Iterate backwards through the layers, except the last layer because we've already calculated its errors.
    for (int layerIndex = layers.size() - 2; layerIndex >= 0; layerIndex--)
    {
      Layer layer = layers.get(layerIndex);

      // For each layer, iterate through each neuron. For each neuron, multiply the errors for the next layer by the
      // weight of the output of the current neuron to each respective next-layer neuron.
      for (int neuronIndex = 0; neuronIndex < layer.getNeuronCount(); neuronIndex++)
      {
        double sumOfErrors = 0d;
        Neuron neuron = layer.getNeurons().get(neuronIndex);
        for (Neuron nextLayerNeuron : layers.get(layerIndex + 1).getNeurons())
        {
          sumOfErrors += nextLayerNeuron.getWeight(neuronIndex) * nextLayerNeuron.getLastError();
        }
        neuron.setLastError(sumOfErrors * layer.getActivatorFunction().calculateDerivative(neuron.getLastOutput()));
      }
    }

    // --------------------------------------------------
    // Update weights
    // --------------------------------------------------

    /*
     * For each hidden layer, iterate through the neurons and update the weight by adding:
     *    For each input:
     *      Learning rate * the error of the current layer neuron * the input
     * Update the hidden layer's bias by adding:
     *    Learning rate * neuron's error
     */

    // Next, update the weights using the same iteration pattern. We can't do this at the same time as the error
    // calculation because that uses the previous weight to calculate error values.
    for(int layerIndex = layers.size() - 1; layerIndex >= 0; layerIndex--)
    {
      Layer layer = layers.get(layerIndex);
      for (Neuron neuron : layer.getNeurons())
      {
        neuron.applyError(learningRate);
      }

    }
  }

  public String printWeights()
  {
    StringBuilder outputBuilder = new StringBuilder();

    for (int i = 0; i < layers.size(); i++)
    {
      outputBuilder.append("Layer ").append(i).append(": ");

      Layer layer = layers.get(i);
      for (Neuron neuron : layer.getNeurons())
      {
        outputBuilder.append("[");
        for(int weightIndex = 0; weightIndex < neuron.getWeightCount(); weightIndex++)
        {
          outputBuilder.append(neuron.getWeight(weightIndex)).append(",");
        }
        outputBuilder.append(neuron.getBias());
        outputBuilder.append("]");
      }

    }
    return outputBuilder.toString();
  }
}
