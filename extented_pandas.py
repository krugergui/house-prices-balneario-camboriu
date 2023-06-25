import pandas as pd
import numpy as np

def infoOut(df: pd.DataFrame, details: bool=False) -> pd.DataFrame:
	dfInfo = df.columns.to_frame(name='Column')
	dfInfo['# Non-Null'] = df.notna().sum()
	dfInfo['Dtype'] = df.dtypes
	dfInfo['% Non-Null'] = dfInfo['# Non-Null'] / len(df)
	dfInfo.reset_index(drop=True,inplace=True)
	dfInfo.index = dfInfo.index + 1

	if details:
		rangeIndex = (dfInfo['# Non-Null'].max(), dfInfo['# Non-Null'].min())
		dtypesCount = dfInfo['Dtype'].value_counts().to_frame()
		dtypesCount.columns = ['# Dtypes']

		print('Max/Min non-null:', rangeIndex, '\n')
		print(dtypesCount, '\n')
	
	dfInfo.style.format({
		'% Non-Null': '{:,.0f}%'.format
	})

	return dfInfo.style.format({
		'% Non-Null': '{:.0%}'.format
	})

def get_lower_and_upper_bounds_for_outliers(series: pd.Series) -> tuple[float, float]:
	"""Receives a panda series and returns the lower and upper bounds that defines the outliers based on the formula:
	Lower bound: Q25 - 1.5 * IQR
	Upper bound: Q75 + 1.5 * IQR

	Args:
		series (pd.Series)
		print_quartiles (bool, optional): Defaults to True.

	Returns:
		tuple[float LowerBound, float UpperBound]
	"""
	q25 = series.quantile(.25)
	q75 = series.quantile(.75)
	iqr = q75 - q25
	lower_q = q25 - (1.5 * iqr)
	upper_q = q75 + (1.5 * iqr)

	return lower_q, upper_q

def AB_Test(dataframe: pd.DataFrame, group: str, target: str, p_val_limit = 0.05) -> pd.DataFrame:
	"""_summary_

	Args:
		dataframe (pd.DataFrame): _description_
		group (str): _description_
		target (str): _description_
		p_val_limit (float, optional): _description_. Defaults to 0.05.

	Returns:
		pd.DataFrame: _description_
	"""
	
	# Packages
	from scipy.stats import shapiro
	import scipy.stats as stats
	
	# Split A/B
	groupA = dataframe[dataframe[group] == 1][target]
	groupB = dataframe[dataframe[group] == 0][target]
	
	# Assumption: Normality
	ntA = shapiro(groupA)[1] < p_val_limit
	ntB = shapiro(groupB)[1] < p_val_limit
	# H0: Distribution is Normal! - False
	# H1: Distribution is not Normal! - True
	
	if (ntA == False) & (ntB == False): # "H0: Normal Distribution"
		# Parametric Test
		# Assumption: Homogeneity of variances
		leveneTest = stats.levene(groupA, groupB)[1] < p_val_limit
		# H0: Homogeneity: False
		# H1: Heterogeneous: True
		
		if leveneTest == False:
			# Homogeneity
			ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
			# H0: M1 == M2 - False
			# H1: M1 != M2 - True
		else:
			# Heterogeneous
			ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
			# H0: M1 == M2 - False
			# H1: M1 != M2 - True
	else:
		# Non-Parametric Test
		ttest = stats.mannwhitneyu(groupA, groupB)[1] 
		# H0: M1 == M2 - False
		# H1: M1 != M2 - True
		
	# Result
	temp = pd.DataFrame({
		"AB Hypothesis":[ttest < p_val_limit], 
		"p-value":[ttest]
	})
	
	temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
	temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
	temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!", "A/B groups are not similar!")
	temp["Feature"] = group
	temp["GroupA_mean"] = groupA.mean()
	temp["GroupB_mean"] = groupB.mean()
	temp["GroupA_median"] = groupA.median()
	temp["GroupB_median"] = groupB.median()
	
	# Columns
	if (ntA == False) & (ntB == False):
		temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
		temp = temp[["Feature","Test Type", "Homogeneity","AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
	else:
		temp = temp[["Feature","Test Type","AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
	
	# Print Hypothesis
	# print("# A/B Testing Hypothesis")
	# print("H0: A == B")
	# print("H1: A != B", "\n")
	
	return temp
