using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.MathUtil.RBF;
using Encog.Neural.NeuralData;
using Encog.Neural.Data.Basic;
using Encog.Neural.Rbf.Training;
using Encog.Neural.RBF;

namespace rbf
{
    class Program
    {
        static void Main(string[] args)
        {
            RbfTester rbfTester = new RbfTester();
            rbfTester.run(10, 40);
            rbfTester.run(10, 1000);
            rbfTester.run(10, 10000);
            rbfTester.run(20, 1000);
            rbfTester.run(50, 10000);
            rbfTester.run(100, 1000);
            //rbfTester.run(10, 1000);
            //rbfTester.run(10, 1000);
        }
    }
}
