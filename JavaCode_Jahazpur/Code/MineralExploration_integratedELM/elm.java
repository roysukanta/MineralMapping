package MineralExploration_ELM;

/*
 * This library is free software;
 * The original version is in Java,I rewrote it  for our application based implementation.
 * The original Authors: MR Dong Li,
 * The original WEBSITE: https://github.com/ExtremeLearningMachines/ELM-JAVA
 * */

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import javax.media.jai.JAI;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.stat.regression.SimpleRegression;












import com.sun.media.jai.codec.TIFFEncodeParam;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix.Norm;
import no.uib.cipr.matrix.NotConvergedException;

public class elm {
	private DenseMatrix train_set;
	private DenseMatrix train_label;
	private DenseMatrix validation_set;
	private DenseMatrix validation_label;
	private DenseMatrix test_set;
	private DenseMatrix test_label;
	private int numTrainDimension;
	private int numvalidationDimension;
	private int numTestData;
	private DenseMatrix InputWeight;
	
	private float TestingTime;
	private float TrainingTime;
	private float validationingTime;
	private double TrainingAccuracy, validationingAccuracy, TestingAccuracy;
	private int Elm_Type;
	private int NumberofHiddenNeurons;
	private int NumberofOutputNeurons;						//also the number of classes
	private int NumberofInputNeurons;						//also the number of attribution
	private String func;
	private int []label;
	
	//the blow variables in both train() and validation()
	private DenseMatrix  BiasofHiddenNeurons;
	private DenseMatrix  OutputWeight;
	private DenseMatrix  validationP;
	private DenseMatrix  validationT;
	private DenseMatrix  Y;
	private DenseMatrix  T;
	private DenseMatrix testT;
	private DenseMatrix testP;
	private double Correlation_threshold;
	
	/**
     * Construct an ELM
     * @param
     * elm_type              - 0 for regression; 1 for (both binary and multi-classes) classification
     * @param
     * numberofHiddenNeurons - Number of hidden neurons assigned to the ELM
     * @param
     * ActivationFunction    - Type of activation function:
     *                      'sig' for Sigmoidal function
     *                      'sin' for Sine function
     *                      'hardlim' for Hardlim function
     *                      'tribas' for Triangular basis function
     *                      'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
	 * @param threshold_corrFact 
     * @throws NotConvergedException
     */
	
	public elm(int elm_type, int numberofHiddenNeurons, String ActivationFunction,int outputNum, double threshold_corrFact){
		
		
		
		Elm_Type = elm_type;
		NumberofHiddenNeurons = numberofHiddenNeurons;
		func = ActivationFunction;
		
		TrainingTime = 0;
		validationingTime = 0;
		TrainingAccuracy= 0;
		validationingAccuracy = 0;
		NumberofOutputNeurons = outputNum;	
		Correlation_threshold = threshold_corrFact;
		
	}

	//the first line of dataset file must be the number of rows and columns,and number of classes if neccessary
	//the first column is the norminal class value 0,1,2...
	//if the class value is 1,2...,number of classes should plus 1
	public DenseMatrix loadmatrix(String filename) throws Exception{
		
	/*	
		BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
		String firstlineString = reader.readLine();
//		String []strings = firstlineString.split(",");
//		int m = Integer.parseInt(strings[0]);
//		int n = Integer.parseInt(strings[1]);		
		
		//DenseMatrix matrix = new DenseMatrix(7138, 205);
		DenseMatrix matrix = new DenseMatrix(5numlibSet000,8);
		
		//firstlineString = reader.readLine();
		int i = 0;
		try{
			while (firstlineString != null) {
				String []datatrings = firstlineString.split(",");
	 			for (int j = 0; j < matrix.numColumns(); j++) {
	 				matrix.set(i, j, Double.parseDouble(datatrings[j]));
				}
				i++;
				firstlineString = reader.readLine();
			}	
		}catch(Exception e){
			System.out.println("after this :"+ i);
		}
		
		
		return matrix;*/
		
		
		
		LineNumberReader lr = new LineNumberReader(new FileReader(new File(filename)));
		lr.skip(Long.MAX_VALUE);
		int numRows = lr.getLineNumber();
		lr.close();
		BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
		String s = reader.readLine();
		int numCols = s.split(",").length;
		DenseMatrix matrix = new DenseMatrix(numRows,numCols);
		System.out.println(numRows+"Here is "+numCols);
		int r=0;
		//s= reader.readLine();
		while (s !=null) {
			String[] data = s.split(",");
			for(int c =0;c<data.length;c++){
				matrix.set(r,c,Double.parseDouble(data[c]));
			}
			s = reader.readLine();
			r++;
		}
	    reader.close();
       
		return matrix;
	    
		
	/*	BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
		String firstlineString = reader.readLine();	
		
		DenseMatrix matrix = new DenseMatrix(7138, 205);
		
		//firstlineString = reader.readLine();
		int i = 0;
		while (firstlineString != null) {
			String []datatrings = firstlineString.split(",");
			for (int j = 0; j < matrix.numColumns(); j++) {
				matrix.set(i, j, Double.parseDouble(datatrings[j]));
			}
			i++;
			firstlineString = reader.readLine();
		}
		return matrix;
		*/
	}
	


	public void train(String TrainingData_File) throws Exception{
		try {
			train_set = loadmatrix(TrainingData_File);
			//train_label = loadmatrix("/home/sukanta/Desktop/3/126_1numlibSetnumlibSetl.csv");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		train();
	}
	
	private void train() throws NotConvergedException{
		
		numTrainDimension = train_set.numRows();
		NumberofInputNeurons = train_set.numColumns() - 1;
		DenseMatrix tempInput = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, NumberofInputNeurons);
		InputWeight = new DenseMatrix(NumberofHiddenNeurons, NumberofInputNeurons);
		for(int r=0;r<NumberofHiddenNeurons;r++){
			for(int c=0;c<NumberofInputNeurons;c++){
			InputWeight.set(r,c,tempInput.get(r,c)*2 -1);	 //to keep input range in between -1 and 1
			}
		}
		
		DenseMatrix transT = new DenseMatrix(numTrainDimension, 1);
		DenseMatrix transP = new DenseMatrix(numTrainDimension, NumberofInputNeurons);
		for (int i = 0; i < numTrainDimension; i++) {
			//transT.set(i,0,train_set.get(i,0));
			transT.set(i, 0, train_set.get(i, 0));
			for (int j = 1; j <= NumberofInputNeurons; j++)
				transP.set(i, j-1, train_set.get(i, j));
		}
		T = new DenseMatrix(1,numTrainDimension);
		DenseMatrix P = new DenseMatrix(NumberofInputNeurons,numTrainDimension);
		transT.transpose(T);
		transP.transpose(P);
		
		HashMap<Integer, Integer> classCount = getCountOfClasses(transT);
		System.out.println(classCount);
		
		if(Elm_Type != 0)	//CLASSIFIER
		{
			label = new int[NumberofOutputNeurons];
			for (int i = 0; i < NumberofOutputNeurons; i++) {
				label[i] = i;							//class label starts form 0
			}
			DenseMatrix tempT = new DenseMatrix(NumberofOutputNeurons,numTrainDimension);
			tempT.zero();
			for (int i = 0; i < numTrainDimension; i++){
					int j = 0;
			        for (j = 0; j < NumberofOutputNeurons; j++){
			            if (label[j] == T.get(0, i))
			                break; 
			        }
			        tempT.set(j, i, 1); 
			}
			
			T = new DenseMatrix(NumberofOutputNeurons,numTrainDimension);	// T=temp_T*2-1;
			for (int i = 0; i < NumberofOutputNeurons; i++){
		        for (int j = 0; j < numTrainDimension; j++)
		        	T.set(i, j, tempT.get(i, j)*2-1);
			}
			
			transT = new DenseMatrix(numTrainDimension,NumberofOutputNeurons);
			T.transpose(transT);
			
		} 	//end if CLASSIFIER
		
		long start_time_train = System.currentTimeMillis();
		// Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
		// InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
		
		BiasofHiddenNeurons = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, 1);
		
		DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons, numTrainDimension);
		InputWeight.mult(P, tempH);
		//DenseMatrix ind = new DenseMatrix(1, numTrainDimension);
		
		DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons, numTrainDimension);
		
		for (int j = 0; j < numTrainDimension; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
			}
		}
	
		tempH.add(BiasMatrix);
		DenseMatrix H = new DenseMatrix(NumberofHiddenNeurons, numTrainDimension);
		
		if(func.startsWith("sig")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTrainDimension; i++) {
					double temp = tempH.get(j, i);
					temp = 1.0f/ (1 + Math.exp(-temp));
					H.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("sin")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTrainDimension; i++) {
					double temp = tempH.get(j, i);
					temp = Math.sin(temp);
					H.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("hardlim")){
			//If you need it ,you can absolutely complete it yourself
		}
		else if(func.startsWith("tribas")){
			//If you need it ,you can absolutely complete it yourself	
		}
		else if(func.startsWith("radbas")){
			//If you need it ,you can absolutely complete it yourself
		}

		DenseMatrix Ht = new DenseMatrix(numTrainDimension,NumberofHiddenNeurons);
		H.transpose(Ht);
		Inverse invers = new Inverse(Ht);
		
		DenseMatrix pinvHt = invers.getMPInverse(0.0000009); 
		OutputWeight = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
	
		pinvHt.mult(transT, OutputWeight);
		
		long end_time_train = System.currentTimeMillis();
		TrainingTime = (end_time_train - start_time_train)*1.0f/1000;
		
		DenseMatrix Yt = new DenseMatrix(numTrainDimension,NumberofOutputNeurons);
		Ht.mult(OutputWeight,Yt);
		Y = new DenseMatrix(NumberofOutputNeurons,numTrainDimension);
		Yt.transpose(Y);
		
		if(Elm_Type == 0){
			double MSE = 0;
			for (int i = 0; i < numTrainDimension; i++) {
				MSE += (Yt.get(i, 0) - transT.get(i, 0))*(Yt.get(i, 0) - transT.get(i, 0));
			}
			TrainingAccuracy = Math.sqrt(MSE/numTrainDimension);
		}
		
		//CLASSIFIER
		else if(Elm_Type == 1){
			float MissClassificationRate_Training=0;
			HashMap<Integer, Integer> missClassified = new HashMap<Integer, Integer>(); 
		    
		    for (int i = 0; i < numTrainDimension; i++) {
				double maxtag1 = Y.get(0, i);
				int tag1 = 0;
				double maxtag2 = T.get(0, i);
				int tag2 = 0;
		    	for (int j = 1; j < NumberofOutputNeurons; j++) {
					if(Y.get(j, i) > maxtag1){
						maxtag1 = Y.get(j, i);
						tag1 = j;
					}
					if(T.get(j, i) > maxtag2){
						maxtag2 = T.get(j, i);
						tag2 = j;
					}
				}
		    	if(tag1 != tag2){
		    		if(missClassified.containsKey(tag2)){
                        missClassified.put(tag2, missClassified.get(tag2) + 1);
                    }else{
                        missClassified.put(tag2, 1);
                    } 
		    		MissClassificationRate_Training ++;
		    	}
		    		
			}
		    System.out.println(missClassified);
		    
		   
		    
		    ///If, each class instance is misclassified, then only use the next commented portion/////
            /*for(int i=0;i< missClassified.size();i++){
                double train = 1- missClassified.get(i)*1.0f/classCount.get(i);
                System.out.println("Training accuracy for class "+(i)+ " : "+ train);     
            } */
		    
		    
		    
		    TrainingAccuracy = 1 - MissClassificationRate_Training*1.0f/numTrainDimension;
			
		}
		
	}
	
	 private HashMap<Integer, Integer> getCountOfClasses(DenseMatrix label) {
	        double[] groups = label.getData();
	        HashMap<Integer, Integer> repetitions = new HashMap<Integer, Integer>();
	 
	        for (int i = 0; i < groups.length; ++i) {
	            int item = (int) groups[i];
	            if (repetitions.containsKey(item))
	                repetitions.put(item, repetitions.get(item) + 1);
	            else
	                repetitions.put(item, 1);
	        }
	 
	        return repetitions;
	    } 
	 
	 public void validation(String validationingData_File) throws Exception{
			
			try {
				validation_set = loadmatrix(validationingData_File);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			numvalidationDimension = validation_set.numRows();
			DenseMatrix tvalidationT = new DenseMatrix(numvalidationDimension, 1);
			DenseMatrix tvalidationP = new DenseMatrix(numvalidationDimension, NumberofInputNeurons);
			for (int i = 0; i < numvalidationDimension; i++) {
				tvalidationT.set(i, 0, validation_set.get(i, 0));
				for (int j = 1; j <= NumberofInputNeurons; j++)
					tvalidationP.set(i, j-1, validation_set.get(i, j));
			}
			
			validationT = new DenseMatrix(1,numvalidationDimension);
			validationP = new DenseMatrix(NumberofInputNeurons,numvalidationDimension);
			tvalidationT.transpose(validationT);
			tvalidationP.transpose(validationP);
			HashMap<Integer, Integer> classCount = getCountOfClasses(tvalidationT);
			System.out.println("validation");
			System.out.println(classCount);
			
			long start_time_validation = System.currentTimeMillis();
				
			DenseMatrix tempH_validation = new DenseMatrix(NumberofHiddenNeurons, numvalidationDimension);
			InputWeight.mult(validationP, tempH_validation);
			DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numvalidationDimension);
			
			for (int j = 0; j < numvalidationDimension; j++) {
				for (int i = 0; i < NumberofHiddenNeurons; i++) {
					BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
				}
			}
		
			tempH_validation.add(BiasMatrix2);
			DenseMatrix H_validation = new DenseMatrix(NumberofHiddenNeurons, numvalidationDimension);
			
			if(func.startsWith("sig")){
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numvalidationDimension; i++) {
						double temp = tempH_validation.get(j, i);
						temp = 1.0f/ (1 + Math.exp(-temp));
						H_validation.set(j, i, temp);
					}
				}
			}
			else if(func.startsWith("sin")){
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numvalidationDimension; i++) {
						double temp = tempH_validation.get(j, i);
						temp = Math.sin(temp);
						H_validation.set(j, i, temp);
					}
				}
			}
			else if(func.startsWith("hardlim")){
				
			}
			else if(func.startsWith("tribas")){
		
			}
			else if(func.startsWith("radbas")){
				
			}
			
			DenseMatrix transH_validation = new DenseMatrix(numvalidationDimension,NumberofHiddenNeurons);
			H_validation.transpose(transH_validation);
			
			
			
			/////Apply estimated output weight beta from training phase here
			DenseMatrix Yout = new DenseMatrix(numvalidationDimension,NumberofOutputNeurons);
			transH_validation.mult(OutputWeight,Yout);
			
			
			
			DenseMatrix validationY = new DenseMatrix(NumberofOutputNeurons,numvalidationDimension);
			Yout.transpose(validationY);
			
			long end_time_validation = System.currentTimeMillis();
			validationingTime = (end_time_validation - start_time_validation)*1.0f/1000;
			
			//REGRESSION
			if(Elm_Type == 0){
				double MSE = 0;
				for (int i = 0; i < numvalidationDimension; i++) {
					MSE += (Yout.get(i, 0) - validationT.get(0,i))*(Yout.get(i, 0) - validationT.get(0,i));
				}
				validationingAccuracy = Math.sqrt(MSE/numvalidationDimension);
			}
			
			
			//CLASSIFIER
			else if(Elm_Type == 1){

				DenseMatrix tempvalidationT = new DenseMatrix(NumberofOutputNeurons,numvalidationDimension);
				for (int i = 0; i < numvalidationDimension; i++){
						int j = 0;
				        for (j = 0; j < NumberofOutputNeurons; j++){
				            if (label[j] == validationT.get(0, i))
				                break; 
				        }
				        tempvalidationT.set(j, i, 1); 
				}
				
				validationT = new DenseMatrix(NumberofOutputNeurons,numvalidationDimension);	
				for (int i = 0; i < NumberofOutputNeurons; i++){
			        for (int j = 0; j < numvalidationDimension; j++)
			        	validationT.set(i, j, tempvalidationT.get(i, j)*2-1);
				}

			    float MissClassificationRate_validationing=0;
				HashMap<Integer, Integer> missClassified = new HashMap<Integer, Integer>(); 

			    for (int i = 0; i < numvalidationDimension; i++) {
					double maxtag1 = validationY.get(0, i);
					int tag1 = 0;
					double maxtag2 = validationT.get(0, i);
					int tag2 = 0;
			    	for (int j = 1; j < NumberofOutputNeurons; j++) {
						if(validationY.get(j, i) > maxtag1){
							maxtag1 = validationY.get(j, i);
							tag1 = j;
						}
						if(validationT.get(j, i) > maxtag2){
							maxtag2 = validationT.get(j, i);
							tag2 = j;
						}
					}
			    	if(tag1 != tag2){
			    		if(missClassified.containsKey(tag2)){
	                        missClassified.put(tag2, missClassified.get(tag2) + 1);
	                    }else{
	                        missClassified.put(tag2, 1);
	                    } 
			    		MissClassificationRate_validationing ++;
			    	}
			    		
				}
			    System.out.println(missClassified);
			    
			   
			    ///Apply the next commented code, when each class instance has miss classified result///////
	            /*for(int i=0;i< missClassified.size();i++){
	                double train = 1- missClassified.get(i)*1.0f/classCount.get(i);
	                System.out.println("validationing accuracy for class "+i+ " : "+ train);     
	            } */
			    
			    
			    validationingAccuracy = 1 - MissClassificationRate_validationing*1.0f/numvalidationDimension;
			    
			}
		}
		
	 
	 
	 
	 
	 public void test(String TestingData_File,int width,int height, String endmemberSignature_File,int NumEndmembers,int DimSignature,String waveFile)
			 throws Exception{
			
			test_set = loadtestmatrix(TestingData_File);
			numTestData = test_set.numRows();
			System.out.println("Number dim"+NumberofInputNeurons);
			//DenseMatrix ttestT = new DenseMatrix(numTestData, 1);
			DenseMatrix ttestP = new DenseMatrix(numTestData, NumberofInputNeurons);
			for (int i = 0; i < numTestData; i++) {
				//ttestT.set(i, 0, test_set.get(i, 0)-1);
				for (int j = 0; j <NumberofInputNeurons; j++)
					ttestP.set(i, j, test_set.get(i, j));
			}
			
			DenseMatrix testT = new DenseMatrix(1,numTestData);
			testP = new DenseMatrix(NumberofInputNeurons,numTestData);
			//ttestT.transpose(testT);
			ttestP.transpose(testP);
			System.gc();
			
			
			
			//////////////////////////////////######################################///////////////////////////////////////
			long start_time_test = System.currentTimeMillis();
			DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
			InputWeight.mult(testP, tempH_test);
			DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numTestData);
			
			for (int j = 0; j < numTestData; j++) {
				for (int i = 0; i < NumberofHiddenNeurons; i++) {
					BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
				}
			}
		
			tempH_test.add(BiasMatrix2);
			DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
			
			if(func.startsWith("sig")){
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTestData; i++) {
						double temp = tempH_test.get(j, i);
						temp = 1.0f/ (1 + Math.exp(-temp));
						H_test.set(j, i, temp);
					}
				}
			}
			else if(func.startsWith("sin")){
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTestData; i++) {
						double temp = tempH_test.get(j, i);
						temp = Math.sin(temp);
						H_test.set(j, i, temp);
					}
				}
			}
			else if(func.startsWith("hardlim")){
				
			}
			else if(func.startsWith("tribas")){
		
			}
			else if(func.startsWith("radbas")){
				
			}
			
			DenseMatrix transH_test = new DenseMatrix(numTestData,NumberofHiddenNeurons);
			H_test.transpose(transH_test);
			DenseMatrix Yout = new DenseMatrix(numTestData,NumberofOutputNeurons);
			transH_test.mult(OutputWeight,Yout);
		
			double[] result = new double[numTestData];
			
			if(Elm_Type == 0){
				for (int i = 0; i < numTestData; i++)
					result[i] = Yout.get(i, 0);
			}
			
			else if(Elm_Type == 1){
				for (int i = 0; i < numTestData; i++) {
					int tagmax = 0;
					double tagvalue = Yout.get(i, 0);
					for (int j = 1; j < NumberofOutputNeurons; j++)
					{
						if(Yout.get(i, j) > tagvalue){
							tagvalue = Yout.get(i, j);
							tagmax = j;
						}
			
					}
					result[i] = tagmax;
					
				}
			}
			///////////////Print classified result out of ELM algorithm///////////////////////////////////////////
			DenseMatrix testLabel = new DenseMatrix(numTestData, 1);
			for(int k=0; k<numTestData; k++){
				testLabel.set(k, 0, result[k]);
			}
				
			HashMap<Integer, Integer> testclassCount = getCountOfClasses(testLabel);
			System.out.println("How many class in test phase by ELM"+testclassCount);
			
			//////////////Write down the classified labels and image from test phase of ELM algorithm////////////// 
			String classified_file =new String(TestingData_File+String.format("_%d_classified",height*width)+".tif");
			clusterImage(result,width,height,classified_file);
			
			////////////###############################End of Testing phase of ELM#################################//////////////////
			
			
			
			
			
			
			///////////########Qualification of classified Result with Spectra Feature Fitting###############///////////////////////
			
			
			///////////////////////////    Read: library of Field Signatures       ////////////////
			
			int numlibSet=NumEndmembers;
			int libDimensions=DimSignature;
			double[][] CRlibSet = new double[numlibSet][libDimensions];
			double [][] libSet = new double[libDimensions][numlibSet];
			String libfile =endmemberSignature_File;
		    LineNumberReader lr = new LineNumberReader(new FileReader(new File(libfile)));
			
		    lr.skip(Long.MAX_VALUE);
		    //int numRows = lr.getLineNumber()-1;
			//System.out.println("numRows"+numRows);
			lr.close();
			   BufferedReader reader = new BufferedReader(new FileReader(new File(libfile)));
			   String s =reader.readLine();
			   s =reader.readLine();
			   int numCols = s.split(",").length;
			  // System.out.println("numcol"+numCols);
			 
			  int r=0;
			  while (s !=null) {
				  
				  String[] data = s.split(",");
				   for(int c =0;c<numlibSet;c++){
					   libSet[r][c]=(Double.parseDouble(data[c]));
				   }
				  
				  s = reader.readLine();
				  r++;
			  }
			  System.out.println(r);
			 reader.close();
			 
			 /////////////////////////                 Check libfile           //////////////
			 
			 for(int i=0;i<numlibSet;i++){
				 System.out.println(libSet[i][1]);
			 }
			 System.out.println("I am done");
			 
			
			 
			 /////////////////////////////    Store file of wavelength      ////////////////////////////
			
			 File wavFile=new File(waveFile);
			 double[] wavelength = new double[libDimensions];
			 try{
			    	BufferedReader reader1 = new BufferedReader(new FileReader(wavFile));
					  
					String  s1 = reader1.readLine();
					   //int numCols = s.split(",").length;
					   //System.out.println("numcol"+numCols);
					 
					int r1=0;
			    	while (s1 !=null) {
			    		wavelength[r1]=(Double.parseDouble(s1));
			   			 s1 = reader1.readLine();
			   			 r1++;
			   		  }
			   		
			   		reader1.close();
			    }catch(FileNotFoundException e){
			          e.printStackTrace();
			    }
		    
		   
			 
			 /////////////////////////////    Perform Continuum Removal Operation   //////////////////////    
			 
			 double[][] libSetR = new double[numlibSet][libDimensions];
			 for(int u =0; u<numlibSet; u++){
					for(int c =0;c<libDimensions;c++){
						libSetR[u][c]=libSet[c][u];
					}
					CRlibSet[u]=HyperspectralToolbox.performContinuumRemoval(libSetR[u], wavelength,libDimensions);
					for(int c =0;c<libDimensions;c++){
						if(Double.isNaN(CRlibSet[u][c])||Double.isInfinite(CRlibSet[u][c])){
							CRlibSet[u][c]=0;
	                    }
					//System.out.println();	
					}
					
				}
				
			
			System.out.println("LIBRARY READ COMPLETED");
			System.out.println("Library Set : " + numlibSet +" "+ libDimensions );
			
			
			////////    Perform the linear regression to find correlation on Spectra feature fitting      //////////
			double[][] testSetR = new double[numTestData][libDimensions];
			double[][] CRtestSet = new double[numTestData][libDimensions];
			BufferedReader reader2 = new BufferedReader(new FileReader(new File(TestingData_File)));
			String s2 = reader2.readLine();
			
			int r2=0;
			//s= reader.readLine();
			while (s2 !=null) {
				String[] data = s2.split(",");
				for(int c =0;c<data.length;c++){
					testSetR[r2][c]=Double.parseDouble(data[c]);
				}
				s2 = reader2.readLine();
				r2++;
			}
			reader2.close();
			
			for(int k=0; k<numTestData;k++){
				CRtestSet[k]=HyperspectralToolbox.performContinuumRemoval(testSetR [k], wavelength,libDimensions);
				for(int c =0;c<libDimensions;c++){
					if(Double.isNaN(CRtestSet[k][c])||Double.isInfinite(CRtestSet[k][c])){
						CRtestSet[k][c]=0;
                    }
				}
			}
			////////////////////////////////
			///Check accuracy with help of library matching///////
		    HashMap<Integer, Integer> repetitions = new HashMap<Integer, Integer>(); 
		    double[] VerifiedOp_classification = new double[numTestData];
			for(int k=0; k<numTestData;k++){
				int l = (int) result[k];
                                SimpleRegression sr=new SimpleRegression();
				for(int i=0;i<libDimensions;i++) {
					//System.out.println("R u getting data: "+test_set.get(k, i)+" "+CRlibSet[l][i]);                           
					
					sr.addData(CRtestSet[k][i],CRlibSet[l][i]);
				}
				/*double intercept=0,slope=0;
				if(sr.hasIntercept()) {
					intercept=sr.getIntercept();
				}
				slope=sr.getSlope(); */
				//System.out.println("The equations are : ");
				//System.out.println("y = "+slope+" x + "+intercept);
				//System.out.println("correlation"+sr.getR());
			   if(sr.getR()>threshold_corrFact){
				        VerifiedOp_classification[k]=l; 
				                  if (repetitions.containsKey(l))
	 				                  repetitions.put(l, repetitions.get(l) + 1);
	 			                  else
	 				                  repetitions.put(l, 1);
			    }else{
			    	    TreeMap<Integer, Double> corrList = new TreeMap<>();
			        	for(int j=0; j<numlibSet;j++) {
			    	      	SimpleRegression sr_loop=new SimpleRegression();
			    		    for(int i=0;i<libDimensions;i++) {
							      sr_loop.addData(CRtestSet[k][i],CRlibSet[j][i]);
						    }
			    		    corrList.put(j, sr_loop.getR());
			    	    }

			    	    TreeMap<Integer, Double> SortedcorrList = (TreeMap<Integer, Double>) reversesortByValues(corrList);
			    	    int label = SortedcorrList.firstKey();
			    	    VerifiedOp_classification[k]=label+11;
			    }
					
			}
			System.out.println("Verified class is showing");
			System.out.println(repetitions);
			System.out.println(repetitions.size());
			for(Entry<Integer, Integer> rep:repetitions.entrySet()){
				System.out.println("critical");
				System.out.println("Class: "+rep.getKey()+"classified: "+rep.getValue()+"Outof"+testclassCount.get(rep.getKey()));
                double train = rep.getValue()*1.0f/testclassCount.get(rep.getKey());
                System.out.println("Training accuracy for class "+(rep.getKey())+ " : "+ train);     
            }

			String verified_resultFile= (TestingData_File+String.format("_%d_verified",height*width)+".tif");
			clusterImage(VerifiedOp_classification,width,height,verified_resultFile);
			
			///////////////////////////////////
		
	 }
	 private DenseMatrix loadtestmatrix(String testingData_File) throws IOException {
			// TODO Auto-generated method stub
			 LineNumberReader lr = new LineNumberReader(new FileReader(new File(testingData_File)));
				lr.skip(Long.MAX_VALUE);
				int numRows = lr.getLineNumber();
				lr.close();
				BufferedReader reader = new BufferedReader(new FileReader(new File(testingData_File)));
				String s = reader.readLine();
				int numCols = s.split(",").length;
				DenseMatrix matrix = new DenseMatrix(numRows,numCols);
				int r=0;
				//s= reader.readLine();
				while (s!=null) {
					String[] data = s.split(",");
					for(int c =0;c<data.length;c++){
						matrix.set(r,c,Double.parseDouble(data[c]));
					}
					s = reader.readLine();
					r++;
				}
			    reader.close();
		        System.out.println("Size");
		        System.out.println("row"+numRows+"col"+numCols);
		       
				return matrix;
			
		}
		 public static <K,V extends Comparable<V>> Map<K,V> reversesortByValues(final Map<K,V> map){
			Comparator<K> valueComparator = new Comparator<K>() {
				public int compare(K k1,K k2){
					int compare = map.get(k2).compareTo(map.get(k1));
					if(compare == 0) return 1;
					else return compare;
				}
			};
			Map<K, V> sortedByValues = new TreeMap<K,V>(valueComparator);
			sortedByValues.putAll(map);
			return sortedByValues;
		}

		public static <K, V extends Comparable<V>> Map<K, V> 
		  sortByValues(final Map<K, V> map) {
		    Comparator<K> valueComparator = 
		             new Comparator<K>() {
		      public int compare(K k1, K k2) {
		        int compare = 
		              map.get(k1).compareTo(map.get(k2));
		        if (compare == 0) 
		          return 1;
		        else 
		          return compare;
		      }
		    };
		 
		    Map<K, V> sortedByValues = 
		      new TreeMap<K, V>(valueComparator);
		    sortedByValues.putAll(map);
		    return sortedByValues;
		    
		 }    
	 public static void clusterImage(double[] result,int width_,int height_,String opfile) throws FileNotFoundException{
	      	
	      	int height = height_;
	      	int width = width_;
	      	System.out.println(height+"Size"+width);
	      	//BufferedImage image = null;
	      	BufferedImage actualImage = null;
	      	File labelOutput = new File("/home/sukanta/Rock-Analysis/Exp_Kaolin_Montmor/Endmember/label.txt");
	      	PrintWriter writer = new PrintWriter(labelOutput);
	      	actualImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	    /*  	try{
	      		BufferedImage ImgBuf=JAI.create("fileload",string).getAsBufferedImage();
	      	    
	      		//image = ImageIO.read(new File("/home/sukanta/Desktop/Check/a/tvdenoisynew_rgb-k_1.jpg"));
	      		height = ImgBuf.getHeight();
	      		width = ImgBuf.getWidth();
	      		
	      		actualImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	      		
	      		
	      	}catch(Exception e){
	      		System.out.println(e);
	      		e.printStackTrace();
	      	}*/
	      	int count = 0;
	      	for(int i =0; i<height; i++){
	      		for(int j =0; j<width;j++){
	      		int cluster =(int)result[i*width+j];
	      		int row;
	      		int column;
	      		if(height > width){
	      			row = count%height;
	      			column = count/height;
	      			
	      		}else if(height < width){
	      			row =count %width; 
	      			column = count/width;
	      			//System.out.println("I am here");
	      		}else{
	      			
	      			row =count%width; 
	      			column = count/width;
	      			
	      			/*row = count/width;
	      			column = count%width;*/
	      		}
	      		//System.out.println(row+"Problem"+column);
	      		if(cluster == 0){
	      			Color c = new Color(0,0,0);            //black
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"black");
	      			//image.setRGB(column, row,c.getRGB());
	      			
	      		}else if(cluster == 1){
	      			Color c = new Color(255,165,0);
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"orange");
	      			//image.setRGB(column, row,c.getRGB());
	      			
	      		}else if(cluster == 2){
	      			Color c = new Color(252,255,0);           //yellow
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"yellow");
	      			//image.setRGB(column, row,c.getRGB());
	      			
	      		}else if(cluster == 3){
	      			Color c = new Color(0,255,255);            //cyan
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"cyan");
	      			//image.setRGB(column, row,c.getRGB());
	      						
	      		}else if(cluster == 4){
	      			Color c = new Color(255,0,255);            //black
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"Magenta");
	      						
	      			
	      		}else if(cluster == 5){ 
	      			Color c = new Color(150,75,0);         //sky blue
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"brown");
	      			//image.setRGB(column, row,c.getRGB());
	      			
	      		  			
	      		}else if(cluster == 6){
	      			Color c = new Color(255,255,255);            //white
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"white");
	      			//image.setRGB(column, row,c.getRGB());s
	      			
	      			
	      		}else if(cluster == 7){
	      			Color c = new Color(143,0,255);            //violet
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"violet");
	      			//image.setRGB(column, row,c.getRGB());
	      			
	      			  			
	      			
	      		}else if(cluster == 8){
	      		//pink
	      			Color c = new Color(0,255,0);           //green
	      			actualImage.setRGB(column, row, c.getRGB());
	      			//image.setRGB(column, row,c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"green");
	      			
	      		}else if(cluster == 9){
	      			Color c = new Color(128,128,128);            //black
	      			actualImage.setRGB(column, row, c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"grey");
	      			//image.setRGB(column, row,c.getRGB());
	      		
	      		     
	      		}else if(cluster == 10){
	      			Color c = new Color(0,0,255);            //blue
	      			actualImage.setRGB(column, row, c.getRGB());
	      			//image.setRGB(column, row,c.getRGB());
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"blue");
	      			
	      			
	      			
	      		}else{
	      			
	      			Color c = new Color(255,0,0);            //red
	      			actualImage.setRGB(column, row, c.getRGB());
	      			//image.setRGB(column, row,c.getRGB());clusterImage
	      			writer.println("Xaxis: "+j+"Yaxis: "+j+"Value: "+"red");
	      		
	      	
	      		}
	      		count++;
	      	 }
	      	}
	      	writer.close();
//	      	System.out.println(fileDest.getAbsolutePath());
//	      	System.exit(0);
	      	TIFFEncodeParam parm=new TIFFEncodeParam();
	        FileOutputStream stream = null;
			try {
				String imagefile= opfile;
				stream = new FileOutputStream(imagefile);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	        javax.media.jai.JAI.create("encode", actualImage,stream,"TIFF",parm);
	      	
	      	
//	      	try{
//	      		ImageIO.write(actualImage, "jpg",fileDest);
//	      	}catch(Exception e){
//	      		e.printStackTrace();
//	      	}
	      	
	    }
	
	public float getTrainingTime() {
		return TrainingTime;
	}
	public double getTrainingAccuracy() {
		return TrainingAccuracy;
	}
	public float getvalidationingTime() {
		return validationingTime;
	}
	public double getvalidationingAccuracy() {
		return validationingAccuracy;
	}
	
	public int getNumberofInputNeurons() {
		return NumberofInputNeurons;
	}
	public int getNumberofHiddenNeurons() {
		return NumberofHiddenNeurons;
	}
	public int getNumberofOutputNeurons() {
		return NumberofOutputNeurons;
	}
	
	public DenseMatrix getInputWeight() {
		return InputWeight;
	}
	
	public DenseMatrix getBiasofHiddenNeurons() {
		return BiasofHiddenNeurons;
	}
	
	public DenseMatrix getOutputWeight() {
		return OutputWeight;
	}

		
public void testWithlabel(String TestingData_File, int numOfSplit, String TestingData_Image
			) throws Exception {
		
	   
	
	   try {
			test_set = loadmatrix(TestingData_File);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		test_set.norm(Norm.Frobenius);
		numTestData = test_set.numRows();
		DenseMatrix ttestT = new DenseMatrix(numTestData, 1);
		DenseMatrix ttestP = new DenseMatrix(numTestData, NumberofInputNeurons);
		for (int i = 0; i < numTestData; i++) {
			ttestT.set(i, 0, test_set.get(i, 0));
			for (int j = 1; j <= NumberofInputNeurons; j++)
				ttestP.set(i, j-1, test_set.get(i, j));
		}
		
		testT = new DenseMatrix(1,numTestData);
		testP = new DenseMatrix(NumberofInputNeurons,numTestData);
		ttestT.transpose(testT);
		ttestP.transpose(testP);
		HashMap<Integer, Integer> classCount = getCountOfClasses(ttestT);
		System.out.println(classCount);
		
		
		//////////////////##############################################//////////////////////////////////
		long start_time_test = System.currentTimeMillis();
		DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		InputWeight.mult(testP, tempH_test);
		DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		
		for (int j = 0; j < numTestData; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
			}
		}
	
		tempH_test.add(BiasMatrix2);
		DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		
		if(func.startsWith("sig")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTestData; i++) {
					double temp = tempH_test.get(j, i);
					temp = 1.0f/ (1 + Math.exp(-temp));
					H_test.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("sin")){
			for (int j = 0; j < NumberofHiddenNeurons; j++) {
				for (int i = 0; i < numTestData; i++) {
					double temp = tempH_test.get(j, i);
					temp = Math.sin(temp);
					H_test.set(j, i, temp);
				}
			}
		}
		else if(func.startsWith("hardlim")){
			
		}
		else if(func.startsWith("tribas")){
	
		}
		else if(func.startsWith("radbas")){
			
		}
		
		DenseMatrix transH_test = new DenseMatrix(numTestData,NumberofHiddenNeurons);
		H_test.transpose(transH_test);
		DenseMatrix Yout = new DenseMatrix(numTestData,NumberofOutputNeurons);
		transH_test.mult(OutputWeight,Yout);
		
		DenseMatrix testY = new DenseMatrix(NumberofOutputNeurons,numTestData);
		Yout.transpose(testY);
		
		long end_time_test = System.currentTimeMillis();
		TestingTime = (end_time_test - start_time_test)*1.0f/1000;

		
		double[]actualresult=new double[numTestData];
		for(int p=0;p<numTestData;p++){
			actualresult[p]=testT.get(0, p);
		//	System.out.println("Actual Result:"+actualresult[p]);
		}
		double[] testresult = new double[numTestData];
		if(Elm_Type == 0){
			for (int i = 0; i < numTestData; i++)
				testresult[i] = Yout.get(i, 0);
			//System.out.println("test Result:"+testresult[i]);
		}
		
		else if(Elm_Type == 1){
			for (int i = 0; i < numTestData; i++) {
				int tagmax = 0;
				double tagvalue = Yout.get(i, 0);
				for (int j = 1; j < NumberofOutputNeurons; j++)
				{
					if(Yout.get(i, j) > tagvalue){
						tagvalue = Yout.get(i, j);
						tagmax = j;
					}
		
				}
				testresult[i] = tagmax;
				//System.out.println("test Result:"+testresult[i]);
			}
		}
		
		
		int uniqueClass=NumberofOutputNeurons;
        ArrayList<int[]> matrix=new ArrayList<int[]>();
        for(double i=0;i<uniqueClass;i++) //loop for predicted values
        {
            int[] row=new int[uniqueClass];
            for(int j=0;j<actualresult.length;j++)
            {
                if(Double.compare(i,testresult[j])==0)
                {
                   for(double k=0;k<uniqueClass;k++)
                   {
                       if(Double.compare(k,actualresult[j])==0)
                           row[(int)k]++;
                   }
                }
            }
            matrix.add(row);
        }
        
        
         System.out.println("Confusion Matrix:");
         System.out.println("Row: Predicted Values");
         System.out.println("Col: Actual Values ");
         for(int[] row:matrix) {
            for(int val:row)
               System.out.print(val+" ");
               System.out.println();
         }
		
		//////////////////////////////////////////////
		
		//REGRESSION
		if(Elm_Type == 0){
			double MSE = 0;
			for (int i = 0; i < numTestData; i++) {
				MSE += (Yout.get(i, 0) - testT.get(0,i))*(Yout.get(i, 0) - testT.get(0,i));
			}
			TestingAccuracy = Math.sqrt(MSE/numTestData);
		}
		
		
		//CLASSIFIER
		else if(Elm_Type == 1){

			DenseMatrix temptestT = new DenseMatrix(NumberofOutputNeurons,numTestData);
			for (int i = 0; i < numTestData; i++){
					int j = 0;
			        for (j = 0; j < NumberofOutputNeurons; j++){
			            if (label[j] == testT.get(0, i))
			                break; 
			        }
			        temptestT.set(j, i, 1); 
			}
			
			testT = new DenseMatrix(NumberofOutputNeurons,numTestData);	
			for (int i = 0; i < NumberofOutputNeurons; i++){
		        for (int j = 0; j < numTestData; j++)
		        	testT.set(i, j, temptestT.get(i, j)*2-1);
			}

		    float MissClassificationRate_Testing=0;
			HashMap<Integer, Integer> missClassified = new HashMap<Integer, Integer>(); 

		    for (int i = 0; i < numTestData; i++) {
				double maxtag1 = testY.get(0, i);
				int tag1 = 0;
				double maxtag2 = testT.get(0, i);
				int tag2 = 0;
		    	for (int j = 1; j < NumberofOutputNeurons; j++) {
					if(testY.get(j, i) > maxtag1){
						maxtag1 = testY.get(j, i);
						tag1 = j;
					}
					if(testT.get(j, i) > maxtag2){
						maxtag2 = testT.get(j, i);
						tag2 = j;
					}
				}
		    	if(tag1 != tag2){
		    		if(missClassified.containsKey(tag2)){
                        missClassified.put(tag2, missClassified.get(tag2) + 1);
                    }else{
                        missClassified.put(tag2, 1);
                    } 
		    		MissClassificationRate_Testing ++;
		    	}
		    		
			}
		    System.out.println(missClassified);
		    ///Apply the next phase of commented code, only when each class has some missclified result//////////
            /*for(int i=0;i< missClassified.size();i++){
                double train = 1- missClassified.get(i)*1.0f/classCount.get(i);
                System.out.println("Testing accuracy for class "+i+ " : "+ train);     
            } */
		    TestingAccuracy = 1 - MissClassificationRate_Testing*1.0f/numTestData;
		    
		}
	}
		
		
		
	
	
}
