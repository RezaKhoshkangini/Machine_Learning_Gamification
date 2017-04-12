
public class MyResultInf {

	private float precision;
	private float recall;
	private float accuracy;
	private float perccorrect;
	private float percincorrect;
	private int TotalCorrect;
	private int TotalInCorrect;
	private int TotalChallenges;
	private int FN, FP, TN, TP;

	public MyResultInf(float precision, float recall, float accuracy, float perccorect, float percincorrect,
			int TotalCorrect, int TotalInCorrect, int TotalChallenges, int FN, int FP, int TN, int TP) {
		// TODO Auto-generated constructor stub
		this.precision = precision;
		this.recall = recall;
		this.accuracy = accuracy;
		this.perccorrect = perccorect;
		this.percincorrect = percincorrect;
		this.TotalCorrect = TotalCorrect;
		this.TotalInCorrect = TotalInCorrect;
		this.TotalChallenges = TotalChallenges;
		this.FN = FN;
		this.FP = FP;
		this.TN = TN;
		this.TP = TP;
	}

	public float getPrc() {
		return precision;
	}

	public float getRec() {
		return recall;
	}

	public float getAcc() {
		return accuracy;
	}

	public float getprccorrect() {
		return perccorrect;
	}

	public float getprcIncorrect() {
		return percincorrect;
	}

	public int gettotalchal() {
		return TotalChallenges;
	}

	public int gettotalCorrect() {
		return TotalCorrect;
	}

	public int gettotalInCorrect() {
		return TotalInCorrect;
	}

	public int getTP() {
		return TP;
	}

	public int getTN() {
		return TN;
	}

	public int getFP() {
		return FP;
	}

	public int getFN() {
		return FN;
	}
}
