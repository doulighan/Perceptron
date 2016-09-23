package doulighan;

import java.util.Random;

/**
 * Resources: https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
 * https://gist.github.com/orhandemirel/666270
 *
 * Created by Devin Oulighan on 9/23/2016.
 *
 * This algorithm is a simple linear classifier.
 * There are 3 features in the data set, x, y, z, which each correspond to values between -10 and 10.
 * 200 points are randomly generated as training data. 100 points of class 1 and 100 of class 0.
 * The points have been written to be linearly separable, otherwise the perceptron would not converge.
 *
 * The perceptron attempts to find a plane which bisects the data into two classes. A summation function takes 3
 * vectors (variable and its weight) as input to compute the output class (1 or 0) through a threshold function.
 * A fourth vector with weight 1 describes the bias. Each iteration through the training data compares the summation
 * function output class to the expected class, then calculates error. The error is then multiplied by the learning
 * rate to shift the weights towards a correct solution. When the error reaches 0, a solution has been found.
 *
 * The weights of each variable correspond to the slope of the solution. Each iteration adjusts the slope of the line
 * to include or exclude data points. The bias (the 4th input vector) has no weight, and simply shifts the equation
 * relative to the origin. (0 = mx + my + mz+ b)
 *
 * After error reaches zero, the equation of the plane is stated, and 10 random test points are generated for the
 * algorithm to classify. This algorithm could potentially be used to check weather a data set is linearly separable
 * or not. It could also be used as a base for a multi-layered perceptron algorithm
 */
public class Perceptron {

  public static int TRAINING_DATA_SIZE = 50;
  public static int ITERATION_LIMIT = 100;     // max iteration if solution is not found
  public static int THETA = 0;                 // adjustable threshold for summation function
  public static double LEARNING_RATE = .1;     // rate of adjustment to the weights

  public static void main(String[] args) {


    double[] x = new double[TRAINING_DATA_SIZE];
    double[] y = new double[TRAINING_DATA_SIZE];
    double[] z = new double[TRAINING_DATA_SIZE];
    int[] outputClasses = new int[TRAINING_DATA_SIZE];  //Will contain either 1 or 0

    double[] weights = new double[4];               //1 weight per each variable and 1 for bias
    double localError;
    double totalError;
    int p;
    int outputClass;

    /////////// Generate training data set ///////////

    //100 of class 1, 100 of class 0, linearly separable
    for (int i = 0; i < TRAINING_DATA_SIZE / 2; i++) {
      x[i] = generateRandomDouble(2, 10);
      y[i] = generateRandomDouble(4, 8);
      z[i] = generateRandomDouble(3, 9);
      outputClasses[i] = 1;
      System.out.println("x: " + x[i] + "\ty: " + y[i] + "\tz: " + z[i] + "\tclass: " + outputClasses[i]);
    }

    for (int i = TRAINING_DATA_SIZE/2; i < TRAINING_DATA_SIZE; i++) {
      x[i] = generateRandomDouble(- 10, 5);
      y[i] = generateRandomDouble(- 3, 4);
      z[i] = generateRandomDouble(- 7, 2);
      outputClasses[i] = 0;
      System.out.println("x: " + x[i] + "\ty: " + y[i] + "\tz: " + z[i] + "\tclass: " + outputClasses[i]);
    }

    weights[0] = generateRandomDouble(0, 1); // weight for x
    weights[1] = generateRandomDouble(0, 1); // weight for y
    weights[2] = generateRandomDouble(0, 1); // weight for z
    weights[3] = generateRandomDouble(0, 1); // bias


    /////////// Train algorithm ///////////

    int iterationCount = 0;
    totalError = -1;        // just to enter the loop without satisfying first condition
    while (totalError != 0 && iterationCount <= ITERATION_LIMIT) {
      totalError = 0;
      iterationCount++;

      //loop through all instances (one complete epoch)
      for (p = 0; p < TRAINING_DATA_SIZE; p++) {

        //summation equation
        double sum = x[p] * weights[0] + y[p] * weights[1] + z[p] * weights[2] + weights[3];
        outputClass = (sum >= THETA) ? 1 : 0;

        // update weights and bias
        localError = outputClasses[p] - outputClass;
        weights[0] += LEARNING_RATE * localError * x[p];
        weights[1] += LEARNING_RATE * localError * y[p];
        weights[2] += LEARNING_RATE * localError * z[p];
        weights[3] += LEARNING_RATE * localError;

        //summation of squared error (error over all instances), for calculating root mean squared error.
        totalError += (localError * localError);
      }
      /* Root Mean Squared Error */
      System.out.println("Iteration " + iterationCount + " : Root Mean Squared = " + Math.sqrt(totalError /
          TRAINING_DATA_SIZE));
    }

    System.out.println
        ("\n==============================================================================================\nDecision " +
            "boundary eqation:");
    System.out.println(weights[0] + "*x + " + weights[1] + "*y + " + weights[2] + "*z + " + weights[3] + " = 0");
    System.out.println
        ("==============================================================================================");


    //Generate 10 new random points and check their classes.
    for (int j = 0; j < 10; j++) {
      double x1 = generateRandomDouble(- 10, 10);
      double y1 = generateRandomDouble(- 10, 10);
      double z1 = generateRandomDouble(- 10, 10);

      double sum = (x1 * weights[0]) + (y1 * weights[1]) + (z1 * weights[2]) + weights[3];
      outputClass = (sum >= THETA) ? 1 : 0;
      System.out.println("\nNew test point:");
      System.out.println("x = " + x1 + ", y = " + y1 + ", z = " + z1);
      System.out.println("class = " + outputClass);
    }

  }

  public static double generateRandomDouble(int rangeMin , int rangeMax) {
    Random r = new Random();
    return rangeMin + (rangeMax - rangeMin) * r.nextDouble();
  }
}
