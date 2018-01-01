using Encog.MathUtil.RBF;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Neural.Data.Basic;
using Encog.Neural.Pattern;
using Encog.Neural.Rbf.Training;
using Encog.Neural.RBF;
using System;

namespace rbf
{
    class RbfTester
    {
        double[][] input;
        double[][] ideal;

        double[][] testInput;
        double[][] testIdeal;

        IMLData tInput;
        IMLData tIdeal;

        double xTrainScale;
        double xTestScale;
        double yTrainScale;
        double yTestScale;

        private BasicNeuralDataSet trainingSet;

        public RbfTester()
        {
        }

        public void run(int neurons, int trainInputCount)
        {

            int dimensions = 1;
            int numNeuronsPerDimension = neurons;
            double volumeNeuronWidth = 2.0 / numNeuronsPerDimension;
            bool includeEdgeRBFs = true;

            var pattern = new RadialBasisPattern();
            pattern.InputNeurons = dimensions;
            pattern.OutputNeurons = 1;
            int numNeurons = (int)Math.Pow(numNeuronsPerDimension, dimensions);
            pattern.AddHiddenLayer(numNeurons);

            var network = (RBFNetwork)pattern.Generate();

            network.SetRBFCentersAndWidthsEqualSpacing(0, 1, RBFEnum.Gaussian, volumeNeuronWidth, includeEdgeRBFs);

            initInputs(trainInputCount, 30);

            IMLDataSet trainingSet = new BasicNeuralDataSet(input, ideal);
            SVDTraining train = new SVDTraining(network, trainingSet);

            int epoch = 1;
            do
            {
                train = new SVDTraining(network, trainingSet);
                train.Iteration();
                Console.WriteLine("Epoch #" + epoch + " Error:" + train.Error);
                epoch++;
            } while ((epoch < 1) && (train.Error > 0.001));

            IMLDataSet testingSet = new BasicMLDataSet(testInput, testIdeal);


            double error = 0.0;
            int count = 0;
            foreach (BasicMLDataPair pair in testingSet)
            {
                BasicMLData output = (BasicMLData)network.Compute(pair.Input);
                double temp = pair.Ideal[0];
                error += Math.Pow(output.Data[0] - temp, 2);
                count++;
            }

            Console.Out.WriteLine(error/count);
        }

        private double f(double x)
        {
            return Math.Sin(Math.Sqrt(x));
        }

        private void initInputs(int trainLen, int testLen)
        {
            input = new double[trainLen][];
            ideal = new double[trainLen][];

            testInput = new double[testLen][];
            testIdeal = new double[testLen][];

            double x = 0.0;
            xTrainScale = x;
            yTrainScale = f(x);
            for (int i = 0; i < trainLen; i++)
            {
                input[i] = new double[1];
                input[i][0] = x;
                ideal[i] = new double[1];
                ideal[i][0] = f(x);

                xTrainScale = (toChange(x, xTrainScale)) ? x : xTrainScale;
                yTrainScale = (toChange(f(x), yTrainScale)) ? f(x) : yTrainScale;

                x += 0.1;
            }

            x = 0.0;
            xTestScale = x;
            yTestScale = f(x);
            for (int i = 0; i < testLen; i++)
            {
                testInput[i] = new double[1];
                testInput[i][0] = x;
                testIdeal[i] = new double[1];
                testIdeal[i][0] = f(x);
                
                xTestScale = (toChange(x, xTestScale)) ? x : xTestScale;
                yTestScale = (toChange(f(x), yTestScale)) ? f(x) : yTestScale;
                x += 0.05;
            }

            for(int i = 0; i< trainLen; i++)
            {
                input[i][0] /= trainLen * 0.1;
                ideal[i][0] /= f(trainLen * 0.1);
            }

            for(int i = 0; i < testLen; i++)
            {
                testInput[i][0] /= testLen * 0.05;
                testIdeal[i][0] /= f(testLen * 0.05);
            }

            //Console.Out.WriteLine(x + ":" + xTestScale);
        }

        private bool toChange(double a, double b)
        {
            if (a < 0)
            {
                if (b < 0)
                    return a < b;
                else
                    return a < (-1 * b);
            } else
            {
                if (b > 0)
                    return a > b;
                else
                    return a > (-1 * b);
            }
        }
    }
}
