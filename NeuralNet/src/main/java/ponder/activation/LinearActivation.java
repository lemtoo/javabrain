package ponder.activation;

public class LinearActivation implements Activation
{
  @Override
  public double activate(double input)
  {
    return input / 10.0d;
  }

  @Override
  public double calculateDerivative(double input)
  {
    return 0.1d;
  }
}
