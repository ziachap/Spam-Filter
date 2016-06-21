package spamFilter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import static java.lang.Math.pow;
import static java.lang.Math.log;

public class filter {

	// GLOBAL variables
	private static String testFile;
	private static List<Word> words = new ArrayList<Word>();
	private static Map<String,Word> vocab = new HashMap<String,Word>();
	private static List<String> triggerWords = new ArrayList<String>();
	private static double spamInstances = 0;
	private static double hamInstances = 0;
	private static double emails = 0;
	private static double spamWeight = 1;
	private static double hamWeight = 1;

	public static void main(String[] args) throws IOException{
		//test = args[0];

		// Load trigger words from file
		loadTriggerWords();

		//OPTIONS
		//sampleTest();
		trainData("train");
		loadTrainingFile();
		//testFile(test);

		boolean bSpam = false;
	  for (int i= 1; i <7; i++) {
	 		File spamFile = new File("train/spam"+ i*12 + ".txt");
			bSpam = testFile(spamFile);
			if(bSpam == true) System.out.println("spam");
			if(bSpam == false) System.out.println("ham");
			File hamFile = new File("train/ham"+ i*9 + ".txt");
			bSpam = testFile(hamFile);
			if(bSpam == true) System.out.println("spam");
			if(bSpam == false) System.out.println("ham");
	  }

		// 10 fold cross validation
		kFold(10, "train");
	}

	private static void trainData(String trainDir) throws IOException{

		// Load all files in dir
		File[] files = new File(trainDir).listFiles();
		emails = (double)files.length;
		System.out.println("Training Data.....");

		//Iterates through files
		for (File file : files) {
			if(file.isFile()){
				trainFile(file);
			} else {
				System.out.println("Not a file");
			}
		}
		// Work out probabilities of each word
		vocabProb();

		System.out.println("Data Trained.....\n");
		System.out.println(spamInstances + "/" + hamInstances);

		storeTrainingData();
	}



	private static void trainFile(File file) throws IOException{
		String[] stopWords = loadStopWords(); //Simple pre-processing technique

			String absolutePath = file.getAbsolutePath();
			String fileString = readFile(absolutePath);
			String[] fileSplit = parseFile(fileString);

			//Sets the type to be spam or ham dependant on file name
			String fileName = file.getName();
			boolean type = fileName.toLowerCase().contains("spam");
			if(type == true) spamInstances ++;
			else hamInstances++;
			//For each word in the file, it enters the add word function
			for (String word : fileSplit){

				//Stop words integrated
				boolean isStopWord = false;
					for (String s : stopWords){
						if (s.equals(word)){
							isStopWord = true;
							break;
						}
					}

				if (isStopWord == false) addWord(word, type);
				//addWord(word, type); //Without Pre-processing
			}


		}


	//Function that tests if a particular email is spam or ham
	private static boolean testFile(File file) throws IOException{
		String absolutePath = file.getAbsolutePath();
		String fileString = readFile(absolutePath);
		checkTriggerWords(fileString, 1.5);
		String[] fileSplit = parseFile(fileString);

		int totalWords = 0;

		//For each word in the file, except from words not in the training vocabulary,
		//it adds one to the number of times that word has been seen already
		for (String word : fileSplit){

			if (vocab.containsKey(word)){
		        Word w = vocab.get(word);
				w.testCount++;
				totalWords++;

			}
		}

		// Work out priors
		double spamPrior = spamInstances/emails;
		double hamPrior = hamInstances/emails;

		//Calculates the likelihoods that is it spam or ham
		double probHam = log(factorial(totalWords));
		double probSpam = log(factorial(totalWords));
		Iterator it = vocab.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        Word w = (Word) pair.getValue();
			probHam += log(pow(w.hamProb, w.testCount) / factorial(w.testCount));
			probSpam += log(pow(w.spamProb, w.testCount) / factorial(w.testCount));

		}

	    it = vocab.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        Word w = (Word) pair.getValue();
			w.testCount = 0;
		}

		// Include prior
	  probSpam = log(spamWeight) + log(spamPrior) + probSpam;
	  probHam = log(hamPrior) + probHam;

		// Calculates the likelihod ratio
	  double likelihoodRatio = probHam - probSpam;

		// Evaluate likelihood ratio
		boolean bSpam;
		if(likelihoodRatio < 0){
			bSpam = true;
		} else {
			bSpam = false;
		}

		// Reset global variables
		spamWeight = 1;

		return bSpam;

	}

	//Parses a string based on whitespace
	private static String[] parseFile(String file){
		String[] fileSplit = file.replaceAll("[^a-zA-Z ]", "").split("\\s+");
		return fileSplit;
	}


	private static void addWord(String word, boolean spam){

		boolean exists = false;
		//Checks if the word being evaluated already exists in the training vocabulary
		//If it does, it just adds 1 to either spam or ham
		if(vocab.containsKey(word)){
			exists = true;
			if(spam) vocab.get(word).spam++;
			else vocab.get(word).ham++;
		}

		//If it is in fact a new word, it will add a new word to the vocabulary
		if(!exists){
			if(spam) vocab.put(word,(new Word(word, 1, 0, 0, 0)));
			else vocab.put(word,(new Word(word, 0, 1, 0, 0)));
		}

	}

	private static String readFile(String file) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(file));
		try {
		    StringBuilder sb = new StringBuilder();
		    String line = br.readLine();

		    while (line != null) {
		        sb.append(line);
		        sb.append(System.lineSeparator());
		        line = br.readLine();
		    }
		    String testText = sb.toString();
		    return testText;

		} finally {
		    br.close();
		}

	}

	private static int factorial(int number){
		if(number <= 1) return 1;
		return number + factorial(number - 1);

	}

	private static void storeTrainingData() throws IOException{

    StringBuilder sb = new StringBuilder();

		// Store priors
		String instances = Double.toString(spamInstances);
		sb.append(instances);
		sb.append(System.getProperty("line.separator"));
		instances = Double.toString(hamInstances);
		sb.append(instances);
		sb.append(System.getProperty("line.separator"));

		// Store words
		Iterator it = vocab.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        Word w = (Word) pair.getValue();
			String vocabWord = w.word + " " + w.spamProb + " " + w.hamProb;
			sb.append(vocabWord);
			sb.append(System.getProperty("line.separator"));
		}


		File file = new File("trainingData.txt");

		// if file doesnt exists, then create it
		if (!file.exists()) {
			file.createNewFile();
		}

		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(sb.toString());
		bw.close();
	}


	private static void loadTrainingFile() throws IOException{
		File file = new File("trainingData.txt");
		String absolutePath = file.getAbsolutePath();
		String fileString = readFile(absolutePath);

		String[] fileSplit = fileString.split("[\\r\\n]+");

		spamInstances = Double.parseDouble(fileSplit[0]);
		hamInstances = Double.parseDouble(fileSplit[1]);
		emails = hamInstances + spamInstances;

		//For each word in the file, it enters the add word function
		for (int i = 2; i < fileSplit.length; i++){
			String [] wordSplit = fileSplit[i].split("\\s+");
			vocab.put(wordSplit[0], (new Word(wordSplit[0], 0, 0, Double.parseDouble(wordSplit[1]), Double.parseDouble(wordSplit[2]))));
		}

		//TOP WORDS
		Map<String,Double> topSpam = new HashMap<String,Double>();
		Map<String,Double> topHam = new HashMap<String,Double>();

		//Find top words prediciting spam and ham
		Iterator it = vocab.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        Word w = (Word) pair.getValue();
			if (topHam.size() < 20) {
				topHam.put(w.word, w.hamProb);
			} else {
				double smallestValue = 1;
				String smallestKey ="";
				for (String key : topHam.keySet()){
					double value = topHam.get(key);
					if (value< smallestValue){
						smallestValue = value;
						smallestKey = key;
					}
				}
				if(w.hamProb > smallestValue){
					topHam.remove(smallestKey);
					topHam.put(w.word, w.hamProb);
				}
			}

			if (topSpam.size() < 20) {
				topSpam.put(w.word, w.spamProb);
			} else {
				double smallestValue = 1;
				String smallestKey ="";
				for (String key : topSpam.keySet()){
					double value = topSpam.get(key);
					if (value< smallestValue){
						smallestValue = value;
						smallestKey = key;
					}
				}
				if(w.spamProb > smallestValue){
					topSpam.remove(smallestKey);
					topSpam.put(w.word, w.spamProb);
				}
			}

		}

		System.out.println("----- Top Spam Words ------ ");
		for (String key : topSpam.keySet()){
			double value = topSpam.get(key);
			System.out.println("Word: " + key + ", with probablity:" + value);
		}

		System.out.println("----- Top Ham Words ------ ");
		for (String key : topHam.keySet()){
			double value = topHam.get(key);
			System.out.println("Word: " + key + ", with probablity:" + value);
		}


	}


	private static String[] loadStopWords() throws IOException{
		File file = new File("stopWords.txt");
		String absolutePath = file.getAbsolutePath();
		String fileString = readFile(absolutePath);

		String[] fileSplit = fileString.split("[\\r\\n]+");

		return fileSplit;
	}

	private static void vocabProb(){

		int tSpamWords = 0;
		int tHamWords = 0;

		Iterator it = vocab.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        Word w = (Word) pair.getValue();
	        w.ham++;
			w.spam++;
			tHamWords += w.ham;		//Finds total number of ham words
			tSpamWords += w.spam;	//Finds total number of spam words
	    }

	    it = vocab.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pair = (Map.Entry)it.next();
	        Word w = (Word) pair.getValue();
	    	w.hamProb = (double)w.ham / (double)tHamWords;
			w.spamProb = (double)w.spam / (double)tSpamWords;
	    }


	}

	// Managing trigger word functions
	private static void loadTriggerWords() throws IOException{
		File file = new File("triggerWords.txt");
		String absolutePath = file.getAbsolutePath();
		String fileString = readFile(absolutePath);
		String[] fileSplit = fileString.split("[\\r\\n]+");
		for (String word : fileSplit){
			triggerWords.add(word);
		}
	}
	private static void checkTriggerWords(String testFile, double d) throws IOException{
		boolean isSpam = false;
		for (String word : triggerWords){
			if(testFile.contains(word)){
			 isSpam = true;
			 spamWeight *= d;
		 }
		 if(!isSpam) hamWeight *= 50;
		}
	}

	private static void kFold(int k, String trainDir) throws IOException{

		//Randomly shuffles file list
		File[] files = new File(trainDir).listFiles();
		List <File> tempList = new ArrayList <File>();
		tempList.addAll(Arrays.asList(files));
		Collections.shuffle(tempList);

		loadStopWords();

		int testSize = files.length/k;
		int testStart = 0;
		boolean bSpam = false;
		boolean fileName = false;

		vocab.clear();

		List<Double> values = new ArrayList<Double>();
		List<Double> fPositiveRate = new ArrayList<Double>();
		List<Double> fNegativeRate = new ArrayList<Double>();
		List<Double> fPositives = new ArrayList<Double>();
		List<Double> fNegatives = new ArrayList<Double>();
		List<Double> totalPos = new ArrayList<Double>();
		List<Double> totalNeg = new ArrayList<Double>();

		for (int j = 0; j < k; j++){
			System.out.println("------Fold: " + j + "-------");

			int trueT=0;
			double falseP = 0;
			double falseN = 0;
			double neg = 0;
			double pos = 0;
			// Training k-1 folds

			if (testStart > 1){
				System.out.println("Training range: 0 To " + testStart);
				for (int r = 0; r < testStart; r++){
					trainFile(tempList.get(r));
				}
			}

			System.out.println("Training range: " + (testStart+testSize) + " To " + tempList.size());
			for (int i = (testStart + testSize); i < tempList.size(); i++){
				trainFile(tempList.get(i));
			}
			vocabProb();
			System.out.println("Testing range:" + testStart + " To " + (testSize+testStart));
			for (int l = testStart; l < testSize+testStart; l++){
				bSpam = testFile(tempList.get(l));
				fileName = tempList.get(l).getName().toLowerCase().contains("spam");
				//System.out.println(bSpam + "," + tempList.get(l).getName() );
				//if(bSpam == true && fileName == true)
				//System.out.println(tempList.get(l).getName() + ": " + bSpam + "/" + fileName);
				if (bSpam == fileName) trueT++;
				if (fileName == true && bSpam == false) falseP++;
				if (fileName == false && bSpam == true) falseN++;
				if (fileName == true) pos++;
				if (fileName == false) neg++;
				//if(bSpam != fileName) System.out.println(tempList.get(l).getName());
				//System.out.println(trueT);

			}

			testStart += testSize;
			System.out.println(trueT + "/" + testSize);
			double acc = (double)trueT/(double)testSize;
			double fpr = falseP/neg;

			System.out.println(falseN);

			double fnr  = falseN/pos;
			values.add(acc);
			fPositiveRate.add(fpr);
			fNegativeRate.add(fnr);
			fPositives.add(falseP);
			fNegatives.add(falseN);
			totalPos.add(pos);
			totalNeg.add(neg);

			vocab.clear();

			System.out.println("Fold: " + j  +", Accuracy:" + acc);

		}

		// Calculate and print error values
		double accTotal = 0;
		double fprTotal = 0;
		double fnrTotal = 0;
		double fpTotal = 0;
		double fnTotal = 0;
		double pos = 0;
		double neg = 0;

		for (int i = 0; i < values.size(); i++){
			accTotal += values.get(i);
			fprTotal += fPositiveRate.get(i);
			fnrTotal += fNegativeRate.get(i);
			fpTotal += fPositives.get(i);
			fnTotal += fNegatives.get(i);
			pos += totalPos.get(i);
			neg += totalNeg.get(i);
		}

		double tpr = 1 - fnrTotal/k;
		double tnr = 1 - fprTotal/k;

		System.out.println("Average Accuracy =" + accTotal/k );
		System.out.println("Average False Positive Rate =" + fprTotal/k );
		System.out.println("Average False Positives =" + fpTotal/k );
		System.out.println("Average False Negative Rate =" + fnrTotal/k );
		System.out.println("Average False Negatives =" + fnTotal/k );
		System.out.println("True Positive Rate =" + tpr );
		System.out.println("True Negative Rate =" + tnr );
		System.out.println("Weighted Average = " + (((pos/k)*tpr)+(tnr*(neg/k))));

	}
}
