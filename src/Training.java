
import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.NBTree;
import weka.core.Instances;

public class Training {

	private static final String Prediction = null;
	private static Object trainset;
	private static Object predictionResult;
	String[] entries = null;
	private Object ReadFiles;
	private Object readerTest;

	final String lineSep = System.getProperty("line.separator");
	private Object tt2;

	public Training() {

	}

	public static void main(String[] args) throws Exception {
		// public static void main(String[] args, Object Resulttest, Object tt)
		// throws Exception {

		Training training = new Training();
		Object Resulttest = null;
		System.out.print("Naive Bayes:");
		training.execute(args, Resulttest, trainset, true, false, false, false);
		System.out.print("Decision Tree:");
		training.execute(args, Resulttest, trainset, false, true, false, false);
		System.out.print("Bayes Network:");
		training.execute(args, Resulttest, trainset, false, false, true, false);
		System.out.print("NaiveBayes Decision Tree:");
		training.execute(args, Resulttest, trainset, false, false, false, true);
	}

	public void execute(String[] args, Object Resulttest, Object tt, boolean nbf, boolean dtf, boolean bnf, boolean nbt)
			throws Exception {
		BufferedReader loader = null;

		// Loading the main dataset for training
		final File dir = new File("/Users/rezakhoshkangini/Documents/CH/Arff->10CH");
		File[] listFiles = dir.listFiles(new FileFilter() {

			@Override
			public boolean accept(File pathname) {
				return pathname.isFile() && pathname.getName().endsWith(".arff");
			}
		});

		// ArrayList<Double> arrlistprediction=new ArrayList<Double>(8);

		ArrayList<Double> arrlistpercision = new ArrayList<Double>(listFiles.length);
		ArrayList<Double> arrlistfmeasure = new ArrayList<Double>(listFiles.length);

		ArrayList<Integer> arrlistTP = new ArrayList<Integer>(listFiles.length);
		ArrayList<Integer> arrlistTN = new ArrayList<Integer>(listFiles.length);

		ArrayList<Integer> arrlistFP = new ArrayList<Integer>(listFiles.length);
		ArrayList<Integer> arrlistFN = new ArrayList<Integer>(listFiles.length);

		ArrayList<Integer> arrlistInstancesCorrect = new ArrayList<Integer>(listFiles.length);
		ArrayList<Integer> arrlistInstancesInCorrect = new ArrayList<Integer>(listFiles.length);

		for (final File fileEntry : listFiles) {

			// System.out.println("User: " + fileEntry.getName() + "\n");
			loader = new BufferedReader(new FileReader(fileEntry));
			Instances train = new Instances(loader);
			// loader.setSource(fileEntry);
			train.setClassIndex(train.numAttributes() - 1);
			loader.close();
			String filename = fileEntry.getName();

			if (nbf) {
				trainset = NaiveBayesfunction(train);

				predictionResult = prediction(trainset, filename);

				// assignper(predictionResult);

			}
			if (dtf) {

				trainset = DecisionTreefunction(train);

			}
			if (bnf) {

				trainset = BayesNetworkfunction(train);

			}
			if (nbt) {

				trainset = NaiveBayesTreefunction(train);
			}
			// Testing Using Cross Validation

			Evaluation eval = new Evaluation(train);

			Resulttest = testcrossValidation(trainset, eval, train);

			// ROC Curve

			// ROC
			// Adding percision into a listArray
			arrlistpercision.add(eval.precision(1));
			arrlistfmeasure.add(eval.fMeasure(1));

			arrlistInstancesCorrect.add((int) eval.correct());
			arrlistInstancesInCorrect.add((int) eval.incorrect());

			arrlistFN.add((int) eval.numFalseNegatives(1));
			arrlistFP.add((int) eval.numFalsePositives(1));
			arrlistTN.add((int) eval.numTrueNegatives(1));
			arrlistTP.add((int) eval.numTruePositives(1));
			// System.out.println(arrlistTP);

		}

		MyResultInf result = CalPrcRec(TotalConfusionMatrix(arrlistFN), TotalConfusionMatrix(arrlistFP),
				TotalConfusionMatrix(arrlistTN), TotalConfusionMatrix(arrlistTP),
				TotalConfusionMatrix(arrlistInstancesCorrect), TotalConfusionMatrix(arrlistInstancesInCorrect));

		System.out.println(
				"Total number of Challenges/Instances: " + result.gettotalchal() + " " + "from XXX players\n ");
		System.out.println("Total Confusion Matrix\n");
		System.out.println("TP :" + result.getTP() + " FN:" + result.getFN() + "\n");
		System.out.println("FP:" + result.getFP() + " TN:" + result.getTN() + "\n\n");
		System.out.println("Total Number of Correct classifiction :" + result.gettotalCorrect() + "--> "
				+ result.getprccorrect() + "%");
		System.out.println("Total Number of InCorrect classifiction :" + result.gettotalInCorrect() + "-->"
				+ result.getprcIncorrect() + "%\n");

		System.out.println("Tota Accuracy:" + result.getAcc() + "\n");
		System.out.println("Tota Recall:" + result.getRec() + "\n");
		System.out.println("Tota Precision:" + result.getPrc() + "\n");

	}

	private Object prediction(Object trainset2, String filename) throws Exception {
		// TODO Auto-generated method stub

		// Loading the players week 11&12 history
		final File dirlastweeks = new File("/Users/rezakhoshkangini/Documents/CH/IndvArff11-12FPrd");

		File[] listFilesW1112 = dirlastweeks.listFiles(new FileFilter() {

			@Override
			public boolean accept(File pathname) {
				return pathname.isFile() && pathname.getName().endsWith(".arff");
			}
		});

		for (final File indFile : listFilesW1112) {
			String predictfilename = indFile.getName();
			if (new String(predictfilename).equals(filename)) {

				Instances test = new Instances(new FileReader(indFile));
				double[][] pridctionsResult = new double[4][2];
				indFile.exists();

				test.setClassIndex(0);

				test.setClassIndex(test.numAttributes() - 1);

				// Predict distribution of instance
				for (int i = 0; i < test.numInstances(); i++) {

					double[] fDistribution = ((NaiveBayes) trainset2).distributionForInstance(test.instance(i));
					pridctionsResult[i][0] = fDistribution[0];
					pridctionsResult[i][1] = fDistribution[1];

					System.out.println("CHallnge: " + i + "  Prediction to Fail: " + roundDouble(fDistribution[0]) + " "
							+ ",Prediction to complete: " + roundDouble(fDistribution[1]) + "\n");

				}
				AddCoumn(pridctionsResult, filename);
				return pridctionsResult;

			}

		}
		return null;

		// for (int j=0;j<listFilesW1112.length;j++){
		// String predictfilename=listFilesW1112[j].getName();
		// if (predictfilename==filename){
		// Instances test = new Instances(listFilesW1112[j]);
		//
		// listFilesW1112[j].close();
		// }
		// }

		// reading arff file to predict the success of fail
		// BufferedReader readerTest = new BufferedReader(
		// new
		// FileReader("/Users/rezakhoshkangini/Documents/CH/IndivTestArff/test1667.arff"));

		// double[][] pridctionsResult = new double[8][2];

		// Instances test = new Instances(readerTest);
		// readerTest.close();

		// test.setClassIndex(test.numAttributes() - 1);
		//
		// // Predict distribution of instance
		// for (int i = 0; i < test.numInstances(); i++) {
		//
		// double[] fDistribution = ((NaiveBayes)
		// trainset2).distributionForInstance(test.instance(i));
		// pridctionsResult[i][0] = fDistribution[0];
		// pridctionsResult[i][1] = fDistribution[1];
		//
		// System.out.println("CHallnge: " + i + " Prediction to Fail: " +
		// roundDouble(fDistribution[0]) + " "
		// + ",Prediction to complete: " + roundDouble(fDistribution[1]) +
		// "\n");
		//
		// }
		// AddCoumn(pridctionsResult);
		// return pridctionsResult;
	}

	private void AddCoumn(double[][] pridctionsResult2, String CSVname) {
		// TODO Auto-generated method stub
		BufferedReader bufferedReader = null;
		PrintWriter p = null;
		String sep = ",";
		String newColF = "Fail_Prob";
		String newColT = "Succ_Prob";
		String[] arryname = CSVname.split(".arff");
		String filename = arryname[0].trim();
		try {

			final File dirCSvInd = new File("/Users/rezakhoshkangini/Documents/CH/IndvCSV11-12FPrd");

			File[] listCSvInd = dirCSvInd.listFiles(new FileFilter() {

				@Override
				public boolean accept(File pathname) {
					return pathname.isFile() && pathname.getName().endsWith(".csv");
				}
			});

			List<String> input = new ArrayList<String>();
			for (final File indCSvFile : listCSvInd) {

				String csvfilename = indCSvFile.getName();

				if (new String(csvfilename).equals(filename)) {
					bufferedReader = new BufferedReader(new FileReader(indCSvFile));
					String readLine = "";
					while ((readLine = bufferedReader.readLine()) != null) {
						input.add(readLine);
					}

					int numOfRecords = input.size();
					if (numOfRecords > 1) {
						List<String> output = new ArrayList<String>();
						String header = input.get(0) + sep + newColF + sep + newColT;
						output.add(header);

						// calculate relative to current date as months
						for (int i = 1; i < numOfRecords; i++) {
							// I am simply going to get the last column from
							// record
							String row = input.get(i);
							StringBuilder res = new StringBuilder(row);
							// StringBuilder res2 = new StringBuilder(row);
							String[] entries = row.split(sep);
							int length = entries.length;

							if (length > 0) {
								res.append(sep);
								String rec = entries[length - 1];

								double Fval = pridctionsResult2[i - 1][0];
								// double Tval = pridctionsResult2[i - 1][1];
								res.append(Fval);
								// res2.append(Tval);

								output.add(res.toString());
								// output.add(res2.toString());
							}

						}

						// p = new PrintWriter(
						// new
						// FileWriter("/Users/rezakhoshkangini/Documents/CH/IndivTest/1667testper.csv",
						// false));

						p = new PrintWriter(new FileWriter(
								"/Users/rezakhoshkangini/Documents/CH/IndvCSV11-12FPrd/" + filename + ".csv", false));

						// Ouch. Very bad way to handle resources. You should
						// find a
						// better way
						p.print("");
						p.close();
						// Write into file
						// p = new PrintWriter(new
						// FileWriter("/Users/rezakhoshkangini/Documents/CH/IndivTest/1667testper.csv"));
						p = new PrintWriter(new FileWriter(
								"/Users/rezakhoshkangini/Documents/CH/CSVindwithPrediction/" + filename + ".csv"));

						for (String row : output) {
							p.println(row);
						}

					} else {
						System.out.println("No records to process");
					}

				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally { // Close file
			if (p != null) {
				p.close();
			}
			if (bufferedReader != null) {
				try {
					bufferedReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		// read csv file
		// List<String> input = new ArrayList<String>();
		// File inputFile = new
		// File("/Users/rezakhoshkangini/Documents/CH/IndivTest/1667test.csv");

		// bufferedReader = new BufferedReader(new FileReader(inputFile));
		// String readLine = "";
		// while ((readLine = bufferedReader.readLine()) != null) {
		// input.add(readLine);
		// }
		//
		// int numOfRecords = input.size();
		// if (numOfRecords > 1) {
		// List<String> output = new ArrayList<String>();
		// String header = input.get(0) + sep + newColF + sep + newColT;
		// output.add(header);
		//
		// // calculate relative to current date as months
		// for (int i = 1; i < numOfRecords; i++) {
		// // I am simply going to get the last column from record
		// String row = input.get(i);
		// StringBuilder res = new StringBuilder(row);
		// // StringBuilder res2 = new StringBuilder(row);
		// String[] entries = row.split(sep);
		// int length = entries.length;
		//
		// if (length > 0) {
		// res.append(sep);
		// String rec = entries[length - 1];
		//
		// double Fval = pridctionsResult2[i - 1][0];
		// // double Tval = pridctionsResult2[i - 1][1];
		// res.append(Fval);
		// // res2.append(Tval);
		//
		// output.add(res.toString());
		// // output.add(res2.toString());
		// }
		//
		// }
		// p = new PrintWriter(
		// new
		// FileWriter("/Users/rezakhoshkangini/Documents/CH/IndivTest/1667testper.csv",
		// false));
		// // Ouch. Very bad way to handle resources. You should find a
		// // better way
		// p.print("");
		// p.close();
		// // Write into file
		// p = new PrintWriter(new
		// FileWriter("/Users/rezakhoshkangini/Documents/CH/IndivTest/1667testper.csv"));
		// for (String row : output) {
		// p.println(row);
		// }
		//
		// } else {
		// System.out.println("No records to process");
		// }
		//
		// } catch (IOException e) {
		// e.printStackTrace();
		// } finally { // Close file
		// if (p != null) {
		// p.close();
		// }
		// if (bufferedReader != null) {
		// try {
		// bufferedReader.close();
		// } catch (IOException e) {
		// e.printStackTrace();
		// }
		// }
		// }

		System.out.println("Ciao Reza, Assigning is Done ");

	}

	private String roundDouble(double d) {
		// TODO Auto-generated method stub
		DecimalFormat numberFormat = new DecimalFormat("#.###");
		// return numberFormat.format(d);
		return numberFormat.format(d);

	}

	private Object testcrossValidation(Object tt, Evaluation eval, Instances train) throws Exception {
		// TODO Auto-generated method stub
		eval.crossValidateModel((Classifier) tt, train, 10, new Random(1));

		return null;
	}

	private Object NaiveBayesTreefunction(Instances train) throws Exception {
		// TODO Auto-generated method stub
		NBTree nbT = new NBTree();
		nbT.buildClassifier(train);
		return nbT;

	}

	private Object BayesNetworkfunction(Instances train) throws Exception {
		// TODO Auto-generated method stub
		BayesNet bn = new BayesNet();
		bn.buildClassifier(train);
		return bn;
	}

	private Object DecisionTreefunction(Instances train) throws Exception {
		// TODO Auto-generated method stub
		// train Decision tree

		String[] options = new String[1];
		options[0] = "-U";

		J48 j48 = new J48();
		j48.setOptions(options);
		j48.buildClassifier(train);

		return j48;

	}

	public Object NaiveBayesfunction(Instances train) throws Exception {
		// TODO Auto-generated method stub

		// train NaiveBayes
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);

		return nb;

	}

	private MyResultInf CalPrcRec(Integer FN, Integer FP, Integer TN, Integer TP, Integer TotalCorrect,
			Integer TotalInCorrect) {
		// TODO calculating the total recall,accuracy and precision
		float Precision, Recall, Accuracy, perccorect, percincorrect;
		int TotalChallenges = ((int) TotalCorrect + TotalInCorrect);
		perccorect = ((float) TotalCorrect / TotalChallenges);
		percincorrect = ((float) TotalInCorrect / TotalChallenges);

		Precision = ((float) TP / ((int) TP + FP));
		Recall = ((float) TP / ((int) TP + FN));
		Accuracy = ((float) ((int) TP + TN) / ((int) TP + TN + FP + FN));

		return new MyResultInf(Precision, Recall, Accuracy, perccorect, percincorrect, TotalCorrect, TotalInCorrect,
				TotalChallenges, FN, FP, TN, TP);
	}

	private Integer TotalConfusionMatrix(ArrayList<Integer> arrlistCM) {
		// TODO Auto-generated method stub
		Integer currentValue;
		double sum = 0;
		if (arrlistCM == null || arrlistCM.isEmpty()) {
			return 0;
		} else {
			int n = arrlistCM.size();

			for (int i = 0; i < n; i++) {
				currentValue = arrlistCM.get(i).intValue();
				sum += currentValue;

			}

			return (int) sum;

		}

	}

}
