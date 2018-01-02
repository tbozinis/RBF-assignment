﻿using System;
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
            for (int i = 1000; i < 10001; i += 1000)
            {
                rbfTester.run(i, 5000);
            }

            for (int i = 1000; i < 10001; i += 1000)
            {
                rbfTester.run(i, 5000);
            }

            Console.Out.Write("\nPress any key to continue...");
            Console.ReadLine();
        }
    }
}
