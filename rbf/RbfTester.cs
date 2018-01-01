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

        private BasicNeuralDataSet trainingSet;

        public RbfTester()
        {
        }

        public void run()
        {

            int dimensions = 1;
            int numNeuronsPerDimension = 7;
            double volumeNeuronWidth = 2.0 / numNeuronsPerDimension;
            bool includeEdgeRBFs = true;

            var pattern = new RadialBasisPattern();
            pattern.InputNeurons = dimensions;
            pattern.OutputNeurons = 1;
            int numNeurons = (int)Math.Pow(numNeuronsPerDimension, dimensions);
            pattern.AddHiddenLayer(numNeurons);

            var network = (RBFNetwork)pattern.Generate();

            network.SetRBFCentersAndWidthsEqualSpacing(0, 1, RBFEnum.Gaussian, volumeNeuronWidth, includeEdgeRBFs);

            initInputs();

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

            foreach (BasicMLDataPair pair in testingSet)
            {
                BasicMLData output = (BasicMLData)network.Compute(pair.Input);
                //output.Data[0] 
                Console.Out.Write(output.ToString());
                Console.Out.WriteLine(pair.Ideal.ToString());
            }
        }

        private double f(double x)
        {
            return Math.Abs(Math.Sin(Math.Sqrt(x) * x));
        }

        private void initInputs()
        {
            input = new double[10000][];
            ideal = new double[10000][];

            testInput = new double[500][];
            testIdeal = new double[500][];

            double maxInput = 10000 * 0.1;
            double maxIdeal = 1;//f(maxInput);

            double x = 0.0;
            for (int i = 0; i < 10000; i++)
            {
                input[i] = new double[1];
                input[i][0] = x / maxInput;
                ideal[i] = new double[1];
                ideal[i][0] = f(x) / maxIdeal;

                x += 0.1;
            }

            maxInput = 500 * 0.05;
            maxIdeal = 1; // f(maxInput);

            x = 0.0;
            for (int i = 0; i < 500; i++)
            {
                testInput[i] = new double[1];
                testInput[i][0] = x / maxInput;
                testIdeal[i] = new double[1];
                testIdeal[i][0] = f(x) / maxIdeal;
                x += 0.05;
            }
        }
    }
}
