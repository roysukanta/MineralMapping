package MineralExploration_ELM;


import javax.swing.JOptionPane;

public class Main_MineralCharacterize {
	

	public static void main(String[] args) throws Exception {
		
		////####################Define the following parameter#########///////////////////////
		/*////########
		 * 
		 *       Define the parameter of ELM_type = 0 as regression Model or ELM_type = 1 as classification Model.
		 *       Define the parameter of 'ELM_DimensionofHiddenLayer' of ELM method.
		 *       Define the parameter of 'ELM_activationfunction_type' = sig for sigmoid function of ELM method.
		 *       Store the number of class of study area as 'outputNumclass' variable for ELM method.
		 *       Store a threshold value of correlation factor for feature fitting process as 'threshold_corrFact'.
		 *          
		 *       
		 *    Prepare the 'comma' separated traingSet and validationSet in such way that the first column belongs to 
		          label information corresponding the each data set. Store 'trainingfile' and 'validationfile' parameters. 
		               1st column of these data set include the information of label. Dimension:(number_Data+1)Xnumber_bands. 
		            
		 *    Prepare the 'comma' separated  endmember's signature file in same sequence with class assignment in learning
		 *             phase of ELM. Store 'endmemberSignaturefile' with Dimension:number_bandsX(number_Endmembers). 
		 *                                                 Store parametric value of 'Num_SpectralSignatures'.
		 *       
		 *    Store the wavelength list of corresponding data set for each phase testing, validation 
		          and training of ELM. Store 'Wavelengthfile'with dimension: number_bandsX1. Store 'Dimension_Signatures'.
		                                  
		 *    Store the 'comma' separated testData of a Region of Interest through row wise storing of 3-dimensional 
		          data(height-by-width-by-Band) to 2-dimensional data(heightXhieght-by-Band). 
		                                                   Store 'testfile','height_testfile'and 'width_testfile' parameters.                                                  
		                                 
		*/
		
		/* For Demo data
		 * 
		 *        Input
		 *            ELM_type=1;
		 *            ELM_DimensionofHiddenLayer=900;
		 *            ELM_activationfunction_type="sig";
		 *            outputNumClass =7;
		 *            threshold_corrFact=0.75;
		 *            trainingfile ="TrainingSet_Pure_7.csv";
		 *            validationfile ="ValidationSet_Pure_7.csv";//Dimension
		 *            SpectralSignaturesignaturefile="SpectralSignature_ClassInstances_110X7.csv";
		 *            Num_SpectralSignatures=7;
		 *            Dimension_Signatures=110;
		 *            Wavelengthfile="JPL2_wave.txt";
		 *            testfile="DummyROI_200by100XBand.txt"
		 *            height_testfile=200;
		 *            width_testfile=100;
		 *            
		 *            
		 *        Output
		 *            Classified and Verified Segmented Image of Region of Interest.
		 *            
		 *            Accuracy of each class: Bare soil, Dolomite, Kaolinite, Montmorillonite, River Sand, 
		 *            Vegetation land and Cultivation land.
		 *            
		 *            After the operation, select endmembers as Bare soil, Kaolinite, Montmorillonite, River Sand, 
		 *            Vegetation land and Cultivation land setting a threshold value at verification accuracy
		 *            (through correlation factor) as 30% for our 'demo data'.
		 */
		

      	int ELM_type =Integer.parseInt(JOptionPane.showInputDialog("Enter the type of ELM model \n 1 = classification Model"));;
		int ELM_DimensionofHiddenLayer = Integer.parseInt(JOptionPane.showInputDialog("Enter the Dimension of Hidden layer for ELM model(e.g.,900)"));;
		
		String ELM_activationfunction_type ="sig";
		
		int outputNumClass=Integer.parseInt(JOptionPane.showInputDialog("Enter the number of class/signatures \n class instances start from zero"));;;
		double threshold_corrFact = Double.parseDouble(JOptionPane.showInputDialog("Enter the threshold value of correlation factor \n  value range 0 to 1"));
		
		
		String trainingfile="/home/sukanta/Rock-Analysis/Upload_data/TrainingSet_Pure_7.csv";
		
		
		
		String validationfile="/home/sukanta/Rock-Analysis/Upload_data/ValidationSet_Pure_7.csv/";
		
		
		
		String SpectralSignaturesignaturefile="/home/sukanta/Rock-Analysis/Upload_data/SpectralSignature_ClassInstances_110X7.csv";
		int Num_SpectralSignatures = Integer.parseInt(JOptionPane.showInputDialog("Enter the number of class/signatures \\n class instances start from zero"));
		
	
		String Wavelengthfile = "/home/sukanta/Rock-Analysis/Upload_data/JPL2_wave.txt";
		
		int Dimension_Signatures =Integer.parseInt(JOptionPane.showInputDialog("Enter the Dimension_Signatures \\n Dimension_Signatures of demo data=110"));
		
	
		String testfile="/home/sukanta/Rock-Analysis/Upload_data/DummyROI_200by100XBand.txt";
		
		//String testGrayImagefile="/home/sukanta/Rock-Analysis/Upload_data/DummyROI_200X100_Gray.tif";
		int height_testfile = Integer.parseInt(JOptionPane.showInputDialog("Enter the height of testData \\n height of demo data=200"));
		int width_testfile = Integer.parseInt(JOptionPane.showInputDialog("Enter the width of testData \\n width of demo data=100"));
		
		
		
	    	
		///////////////####################################################///////////////////
		
		elm ds = new elm(ELM_type, ELM_DimensionofHiddenLayer,ELM_activationfunction_type,outputNumClass,threshold_corrFact);
		ds.train(trainingfile);
		ds.validation(validationfile);
		System.out.println("TrainingTime: " + ds.getTrainingTime());
		System.out.println("TrainingAcc: " + ds.getTrainingAccuracy());
		System.out.println("ValidationTime: " + ds.getvalidationingTime());
		System.out.println("ValidationAcc: " + ds.getvalidationingAccuracy());
		
        ds.test(testfile,width_testfile,height_testfile,SpectralSignaturesignaturefile,Num_SpectralSignatures,Dimension_Signatures,Wavelengthfile);


	}
}
