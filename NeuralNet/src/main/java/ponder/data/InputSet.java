package ponder.data;

import ponder.err.PonderInvalidInputException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class InputSet
{
  private String[] featureNames;
  private double[][] inputs;

  public InputSet(String sourceCSV) throws PonderInvalidInputException
  {
    try
    {
      try (BufferedReader br = new BufferedReader(new FileReader(sourceCSV)))
      {
        String line;

        // First, read the feature names.
        line = br.readLine();
        featureNames = line.split(",");

        List<List<Double>> featureValues = new ArrayList<>();
        // Read each remaining line into an array of doubles.
        while ((line = br.readLine()) != null)
        {
          String[] values = line.split(",");
          List<Double> currentFeatures = new ArrayList<>();
          for (String value : values)
          {
            currentFeatures.add(Double.parseDouble(value));
          }
          featureValues.add(currentFeatures);
        }

        // Finally, convert the lists to arrays.
        inputs = new double[featureValues.size()][];
        for (int rowIndex = 0; rowIndex < featureValues.size(); rowIndex++)
        {
          List<Double> currentRow = featureValues.get(rowIndex);
          inputs[rowIndex] = new double[currentRow.size()];
          for (int featureIndex = 0; featureIndex < currentRow.size(); featureIndex++)
          {
            Double featureValue = currentRow.get(featureIndex);
            inputs[rowIndex][featureIndex] = featureValue;
          }
        }
      }
    }
    catch (IOException e)
    {
      throw new PonderInvalidInputException("Unable to read input file!");
    }
    catch(NumberFormatException nfe)
    {
      throw new PonderInvalidInputException("Input feature value cannot be interpreted as a double!");
    }
  }

  public String[] getFeatureNames()
  {
    return featureNames;
  }

  public double[][] getInputs()
  {
    return inputs;
  }

  public double[][] getInputFeatures(String[] inputFeatureNames) throws PonderInvalidInputException
  {
    int numRows = inputs.length;

    // Initialize a new input array so that we can filter out the sets we don't want.
    double[][] requestedFeatures = new double[numRows][];
    for(int rowIndex = 0; rowIndex < numRows; rowIndex++)
    {
      requestedFeatures[rowIndex] = new double[inputFeatureNames.length];
    }

    for (int newInputIndex = 0; newInputIndex < inputFeatureNames.length; newInputIndex++)
    {
      int inputFeatureIndex = findFeature(inputFeatureNames[newInputIndex]);

      for(int rowIndex = 0; rowIndex < numRows; rowIndex++)
      {
        requestedFeatures[rowIndex][newInputIndex] = inputs[rowIndex][inputFeatureIndex];
      }
    }

    return requestedFeatures;
  }

  private int findFeature(String inputFeatureName) throws PonderInvalidInputException
  {
    for (int i = 0; i < featureNames.length; i++)
    {
      String featureName = featureNames[i];
      if(featureName.equals(inputFeatureName))
      {
        return i;
      }
    }

    throw new PonderInvalidInputException("Requested feature [" + inputFeatureName +
        "] was not found in the input set!");
  }
}
