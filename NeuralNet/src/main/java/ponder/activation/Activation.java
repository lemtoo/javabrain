package ponder.activation;

public interface Activation
{
  double activate(double input);
  double calculateDerivative(double input);
}
