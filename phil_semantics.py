import sys
import pandas as pd
import numpy as np
import Single_Trial_RSA as STR
from PPData_alt import *
from scipy import stats
from timeit import default_timer as timer
from CPerm import *
import matplotlib.pyplot as plt
import pickle
from RSA_plot import *
import variables_processing_alt as VP
from nltk.corpus import wordnet

# EEG_DIR = "/Users/ted/Dropbox/Phil_RSA/rawdata/Lexical_Decision/"
# EEG_NAME = ['01-KW_LD-AS-ALL.dat','03-KW_LD-YY-ALL.dat','05-KW_LD-JP-ALL.dat','07-KW_LD-KT-ALL.dat','08-KW_LD-MG-ALL.dat','09-KW_LD-EM-ALL.dat','11-KW_LD-ND-ALL.dat','12-KW_LD-KC-ALL.dat','13-KW_LD-LK-ALL.dat','14-KW_LD-ES-ALL.dat','15-KW_LD-AH-ALL.dat','16-KW_LD-NM-ALL.dat','17-KW_LD-DS-ALL.dat','18-KW_LD-AD-ALL.dat','20-KW_LD-FA-ALL.dat','21-KW_LD-PC-ALL.dat','22-KW_LD-SF-ALL.dat','24-KW_LD-RM-ALL.dat','25-KW_LD-NH-ALL.dat','26-KW_LD-TC-ALL.dat','28-KW_LD-BR-ALL.dat','31-KW_LD-EB-ALL.dat','33-KW_LD-SM-ALL.dat','34-KW_LD-MC-ALL.dat','35-KW_LD-JL-ALL.dat','36-KW_LD-MH-ALL.dat','37-KW_LD-RG-ALL.dat','38-KW_LD-JJ-ALL.dat','39-KW_LD-JJ-ALL.dat','40-KW_LD-MM-ALL.dat','41-KW_LD-RS-ALL.dat','42-KW_LD-LG-ALL.dat','43-KW_LD-SS-ALL.dat','44-KW_LD-LM-ALL.dat','45-KW_LD-FF-ALL.dat','46-KW_LD-LK-ALL.dat','48-KW_LD-DR-ALL.dat','49-KW_LD-TM-ALL.dat','50-KW_LD-ET-ALL.dat','51-KW_LD-SK-ALL.dat','53-KW_LD-RN-ALL.dat','54-KW_LD-RL-ALL.dat','55-KW_LD-PM-ALL.dat','56-KW_LD-LC-ALL.dat','57-KW_LD-MO-ALL.dat','58-KW_LD-SC-ALL.dat','59-KW_LD-SS-ALL.dat','60-KW_LD-EC-ALL.dat','61-KW_LD-AF-ALL.dat','62-KW_LD-MD-ALL.dat']

EEG_DIR = "/Users/ted/Dropbox/Phil_RSA/rawdata/Semantic_Categorization/"
EEG_NAME = ['02-KW-OL-ALL.dat','03-KW-CB-ALL.dat','04-kw-mb-ALL.dat','05-KW-AL-ALL.dat','06-KW-AJ-ALL.dat','08-KW-AR-ALL.dat','10-KW-MD-ALL.dat','11-KW-SC-ALL.dat','12-KW-VW-ALL.dat','13-KW-EH-ALL.dat','14-KW-VP-ALL.dat','15-KW-LC-ALL.dat','17-KW-AS-ALL.dat','19-kw-JB-ALL.dat','20-kw-SD-ALL.dat','21-kw-MM-ALL.dat','22-kw-RG-ALL.dat','23-kw-KO-ALL.dat','24-kw-KC-ALL.dat','25-kw-EV-ALL.dat','26-kw-JA-ALL.dat','27-kw-EB-ALL.dat','28-kw-RW-ALL.dat','29-kw-DB-ALL.dat','30-kw-LK-ALL.dat','32-kw-DW-ALL.dat','33-kw-BF-ALL.dat','34-kw-JT-ALL.dat','35-kw-KR-ALL.dat','36-kw-AF-ALL.dat','37-kw-AP-ALL.dat','38-kw-NB-ALL.dat','39-kw-WF-ALL.dat','40-kw-AA-ALL.dat','41-kw-DL-ALL.dat','42-kw-JT-ALL.dat','43-kw-LR-ALL.dat','44-kw-EB-ALL.dat','45-kw-JL-ALL.dat','46-kw-HK-ALL.dat','47-kw-BM-ALL.dat','48-kw-DB-ALL.dat','49-kw-JC-ALL.dat','50-kw-AS-ALL.dat','51-kw-TM-ALL.dat','52-kw-MM-ALL.dat','53-kw-CR-ALL.dat','54-kw-MS-ALL.dat','55-kw-SG-ALL.dat','57-kw-LK-ALL.dat','58-kw-SM-ALL.dat','59-kw-JA-ALL.dat','60-kw-IR-ALL.dat','61-kw-KD-ALL.dat','62-kw-VP-ALL.dat']

RESOURCE_DIRECTORY = "./Resources/"
ALL_VAR = np.array(["wordnet", "wordnet_dom", "path", "w2v", "glove", "lsa", "Concreteness", "Length", "OLD", "Log_Freq"])
ALL_IV = np.array(["wordnet", "wordnet_dom", "path", "w2v", "glove", "lsa", "Concreteness"])
# ALL_IV = np.array(["wordnet", "wordnet_dom", "path", "w2v", "glove", "lsa"])
# ALL_IV = np.array(["path"])
ALL_CV = np.array(["Length", "OLD", "Log_Freq"])

def main():
	start = timer()
	# noun_labels = np.loadtxt(RESOURCE_DIRECTORY+"noun_labels.txt", dtype = str)
	# exact_nouns = noun_labels[:, 0][np.array(noun_labels[:, 1], dtype = bool)]
	# results = assemble_sim_mat(exact_nouns)
	# phil_nouns = load_instance("phil_nouns", RESOURCE_DIRECTORY)
	# phil_nouns.update_variable("wordnet_dom", var_matrix = results)
	# phil_nouns.print_attribute_info()
	# save_instance(phil_nouns, "phil_nouns", RESOURCE_DIRECTORY)

	# phil_nouns = VP.variables(exact_nouns)
	# phil_nouns.update_variable("wordnet", var_matrix = results)
	# phil_nouns.semantic_disimilarilty(exact_nouns, sim_type = "w2v", update = True, resource_directory = RESOURCE_DIRECTORY)
	# phil_nouns.semantic_disimilarilty(exact_nouns, sim_type = "glove", update = True, resource_directory = RESOURCE_DIRECTORY)
	# phil_nouns.print_attribute_info()
	# save_instance(phil_nouns, "phil_nouns", RESOURCE_DIRECTORY)

	# phil_nouns = load_instance("phil_nouns_LSA", RESOURCE_DIRECTORY)
	# all_names = []
	# all_tris = []	
	# for name in ALL_IV:
	# 	curr_name = name
	# 	results = phil_nouns.export_variables(np.array([name]))
	# 	curr_mat = 1 - results["var_matrices"][0]
	# 	curr_tri = 1 - results["var_triangulars"][0]
	# 	all_names.append(curr_name)
	# 	all_tris.append(curr_tri)
		# mean = np.mean(curr_tri)
		# std = np.std(curr_tri)
		# print (curr_name)
		# print (mean)
		# print (std)
		# plot_density(np.array([curr_name]), np.array([curr_tri]), normalize = False, hist = True, show = False, save = True, save_name = "./" + curr_name + "_density.png")
		# plot_matrix(curr_mat, curr_name, clim = (mean - std, mean + std), save = True)
	# plot_density(np.array(all_names), np.array(all_tris), normalize = True, hist = False, save = True)
	# phil_nouns.print_attribute_info()
	# save_instance(phil_nouns, "phil_nouns_LSA", RESOURCE_DIRECTORY)

	# phil_nouns = load_instance("phil_nouns", RESOURCE_DIRECTORY)
	# bws, bwid = phil_nouns.check_missing_words(phil_nouns.words, sim_type = "lsa", resource_directory = RESOURCE_DIRECTORY)
	# print (bws)
	# phil_nouns.delete_trials(indices = bwid)
	# phil_nouns.delete_trials(words = np.array(["governor", "nothing"]))
	# phil_nouns.print_attribute_info()
	# phil_nouns.semantic_disimilarilty(phil_nouns.words, sim_type = "lsa", resource_directory = RESOURCE_DIRECTORY)
	# save_instance(phil_nouns, "phil_nouns_LSA", RESOURCE_DIRECTORY)

	# phil_nouns = load_instance("phil_nouns_LSA", RESOURCE_DIRECTORY)
	# corr_mat = phil_nouns.calculate_variables_correlation(ALL_IV)
	# # corr_mat = phil_nouns.calculate_variables_correlation(ALL_IV, control_var = ALL_CV)
	# print(corr_mat)
	# plot_matrix(corr_mat, "IV dsm corr", var_names = ALL_VAR, clim = (-0.5,0.5), save = True)
	
	# noun_labels = np.loadtxt(RESOURCE_DIRECTORY+"noun_labels.txt", dtype = str)
	# noun_masks = np.array(noun_labels[:, 1], dtype = bool)	
	# phil_nouns = load_instance("phil_nouns", RESOURCE_DIRECTORY)
	# all_var = np.loadtxt(RESOURCE_DIRECTORY+"phil_variables.txt", dtype = str)
	# all_var = all_var[noun_masks, :]
	# print (all_var.shape)
	# concreteness = np.array(all_var[:, 0], dtype = float)
	# length = np.array(all_var[:, 1], dtype = float)
	# OLD = np.array(all_var[:, 2], dtype = float)
	# log_freq = np.array(all_var[:, 3], dtype = float)
	# phil_nouns.update_variable("Concreteness", var_array = concreteness)
	# phil_nouns.update_variable("Length", var_array = length)
	# phil_nouns.update_variable("OLD", var_array = OLD)
	# phil_nouns.update_variable("Log_Freq", var_array = log_freq)
	# phil_nouns.print_attribute_info()
	# save_instance(phil_nouns, "phil_nouns", RESOURCE_DIRECTORY)

	# noun_labels = np.loadtxt(RESOURCE_DIRECTORY+"noun_labels.txt", dtype = str)
	# noun_masks = np.array(noun_labels[:, 1], dtype = bool)
	# print (sum(noun_masks))
	# subjects = load_subjects(noun_masks)
	# save_instance(subjects, "phil_subjects_lexical", RESOURCE_DIRECTORY)

	phil_nouns = load_instance("phil_nouns_LSA", RESOURCE_DIRECTORY)
	phil_nouns.print_attribute_info()
	exit()
	lex_subjects = load_instance("phil_subjects_lexical_60%", RESOURCE_DIRECTORY)
	sem_subjects = load_instance("phil_subjects_semantic_60%", RESOURCE_DIRECTORY)
	# run_lexical(lex_subjects, phil_nouns, path = "./Program_output/" ,save_affix = "_all")
	# run_semantic(sem_subjects, phil_nouns, path = "./Program_output/" ,save_affix = "_all")

	new_lex_subs = []
	for sub in lex_subjects:
		if len(sub.bins) >= 434:
			new_lex_subs.append(sub)
	run_lexical(new_lex_subs, phil_nouns, path = "./Program_output/")
	new_sem_subs = []
	for sub in sem_subjects:
		if len(sub.bins) >= 434:
			new_sem_subs.append(sub)
	run_semantic(new_sem_subs, phil_nouns, path = "./Program_output/")

	# lex_subjects = load_instance("phil_subjects_semantic", RESOURCE_DIRECTORY)
	# lex_subjects = np.array(lex_subjects)
	# mask = np.zeros(len(EEG_NAME), dtype = bool)
	# temp_subs = load_subjects(None)
	# percentage = []
	# count = 0
	# index = 0
	# for sub in temp_subs:
	# 	percentage.append(len(sub.bins)/960)
	# 	if percentage[-1] >= 0.6:
	# 		count += 1
	# 		mask[index] = 1
	# 	index += 1
	# print (count)
	# save_instance(lex_subjects[mask], "phil_subjects_semantic_60%", RESOURCE_DIRECTORY)

	# semantic
	# STR_subs = create_STR(subjects, phil_nouns, ALL_IV, None)
	# all_titles, sem_subs, all_corr = run_STR(STR_subs, ALL_IV, None, "semantic.png")
	# sem_subs = sem_subs[:,:,25:]
	# sem_subs = np.swapaxes(np.array(sem_subs), 0, 1)
	# # lexical
	# STR_subs = create_STR(subjects, phil_nouns, ALL_CV, None)
	# all_titles, lex_subs, all_corr = run_STR(STR_subs, ALL_CV, None, "lexical.png")
	# lex_subs = lex_subs[:,:,25:]
	# lex_subs = np.swapaxes(np.array(lex_subs), 0, 1)
	# semantic partial
	# STR_subs = create_STR(sem_subjects, phil_nouns, ALL_VAR, ALL_CV)
	# all_titles, par_subs_ori, all_corr = run_STR(STR_subs, ALL_IV, ALL_CV, "./Program_output/I_lexical_par_all.png")
	# par_subs = par_subs_ori[:,:,25:]
	# par_subs = np.swapaxes(np.array(par_subs), 0, 1)
	# par_subs_ori = np.swapaxes(np.array(par_subs_ori), 0, 1)
	# print (par_subs_ori.shape)
	# save_instance(par_subs_ori, "lex_subs", "./Save_files/")

	# print ("Semantic results")
	# run_CPerm(sem_subs, ALL_IV, 1000, "./")
	# print ("Lexical results")
	# run_CPerm(lex_subs, ALL_CV, 1000, "./")
	# print ("Partial results")
	# run_CPerm(par_subs, ALL_IV, 10000, "./")

	# par_subs_lex = load_instance("lex_subs", "./Save_files/")
	# par_subs_sem = load_instance("sem_subs", "./Save_files/")
	# print ("lexical results")
	# run_CPerm(par_subs_lex, ALL_IV, "./")
	# print ("\n\n")
	# print ("semantic results")
	# run_CPerm(par_subs_sem, ALL_IV, "./")

	# STR_subs = create_STR_interpol(subjects, phil_nouns, ALL_IV, None)
	# all_subs, all_titles = run_STR_interpol(STR_subs, False, "lexical_par_interpol.png")
	# save_instance(all_subs, "lex_subs_interpol", "./Save_files/")
	# all_subs = all_subs[:,:, 26:]
	# all_subs = np.swapaxes(np.array(all_subs), 0, 1)
	# run_CPerm(all_subs, ALL_IV, "./")

	end = timer()
	print ("execution time: " + str(end - start))
	return

def run_lexical(subjects, phil_nouns, path = "./", save_affix = ""):
	print ("########################\t", end = "")
	print ("Lexical Condition", end = "")
	print ("\t########################")
	# # semantic
	# print ("Analyzing Semantic Variables")
	# STR_subs = create_STR(subjects, phil_nouns, ALL_IV, None)
	# all_titles, sem_subs, all_corr = run_STR(STR_subs, ALL_IV, None, path+"l_semantic"+save_affix+".png")
	# sem_subs = sem_subs[:,:,25:]
	# sem_subs = np.swapaxes(np.array(sem_subs), 0, 1)
	# sem_titles, sem_result = run_CPerm(sem_subs, ALL_IV, 1000, "./")
	# # lexical
	# print ("\nAnalyzing Lexical Variables")
	# STR_subs = create_STR(subjects, phil_nouns, ALL_CV, None)
	# all_titles, lex_subs, all_corr = run_STR(STR_subs, ALL_CV, None, path+"l_lexical"+save_affix+".png")
	# lex_subs = lex_subs[:,:,25:]
	# lex_subs = np.swapaxes(np.array(lex_subs), 0, 1)
	# lex_titles, lex_result = run_CPerm(lex_subs, ALL_CV, 1000, "./")
	# semantic partial
	print ("\nAnalyzing Semantic Variables with partial correlation")
	STR_subs = create_STR(subjects, phil_nouns, ALL_VAR, ALL_CV)
	all_titles, par_subs_ori, all_corr = run_STR(STR_subs, ALL_IV, ALL_CV, path+"l_lexical_par"+save_affix+".png")
	par_subs = par_subs_ori[:,:,25:]
	par_subs = np.swapaxes(np.array(par_subs), 0, 1)
	par_subs_ori = np.swapaxes(np.array(par_subs_ori), 0, 1)
	# sem_par_titles, sem_par_result = run_CPerm(par_subs, ALL_IV, 10000, "./")
	
	# save correlation results
	save_instance(par_subs_ori, "lex_subs"+save_affix, "./Save_files/")
	save_instance(all_corr, "lex_corrs"+save_affix, "./Save_files/")
	# save CPerm results
	# with open (path+"lexical_CPerm"+save_affix+".txt", "w") as outfile:
		# outfile.write("Semantic Variable Result")
		# for index in range(len(sem_result)):
		# 	outfile.write("\n" + sem_titles[index]+":\n")
		# 	sem_result[index].to_string(outfile)
		# outfile.write("\n\n\nLexical Variable Result")
		# for index in range(len(lex_result)):
		# 	outfile.write("\n" + lex_titles[index]+":\n")
		# 	lex_result[index].to_string(outfile)
		# outfile.write("\n\n\nSemantic Variable Partial Correlation Result")
		# for index in range(len(sem_par_result)):
		# 	outfile.write("\n" + sem_par_titles[index]+":\n")
		# 	if type(sem_par_result[index]) is list:
		# 		outfile.write("No significant cluster found")			
		# 	else:
		# 		sem_par_result[index].to_string(outfile)

	return

def run_semantic(subjects, phil_nouns, path = "./", save_affix = ""):
	print ("########################\t", end = "")
	print ("Semantic Condition", end = "")
	print ("\t########################")
	# # semantic
	# print ("Analyzing Semantic Variables")
	# STR_subs = create_STR(subjects, phil_nouns, ALL_IV, None)
	# all_titles, sem_subs, all_corr = run_STR(STR_subs, ALL_IV, None, path+"s_semantic"+save_affix+".png")
	# sem_subs = sem_subs[:,:,25:]
	# sem_subs = np.swapaxes(np.array(sem_subs), 0, 1)
	# sem_titles, sem_result = run_CPerm(sem_subs, ALL_IV, 1000, "./")
	# # lexical
	# print ("\nAnalyzing Lexical Variables")
	# STR_subs = create_STR(subjects, phil_nouns, ALL_CV, None)
	# all_titles, lex_subs, all_corr = run_STR(STR_subs, ALL_CV, None, path+"s_lexical"+save_affix+".png")
	# lex_subs = lex_subs[:,:,25:]
	# lex_subs = np.swapaxes(np.array(lex_subs), 0, 1)
	# lex_titles, lex_result = run_CPerm(lex_subs, ALL_CV, 1000, "./")
	# semantic partial
	print ("\nAnalyzing Semantic Variables with partial correlation")
	STR_subs = create_STR(subjects, phil_nouns, ALL_VAR, ALL_CV)
	all_titles, par_subs_ori, all_corr = run_STR(STR_subs, ALL_IV, ALL_CV, path+"s_semantic_par"+save_affix+".png")
	par_subs = par_subs_ori[:,:,25:]
	par_subs = np.swapaxes(np.array(par_subs), 0, 1)
	par_subs_ori = np.swapaxes(np.array(par_subs_ori), 0, 1)
	# sem_par_titles, sem_par_result = run_CPerm(par_subs, ALL_IV, 10000, "./")
	
	# save correlation results
	save_instance(par_subs_ori, "sem_subs"+save_affix, "./Save_files/")
	save_instance(all_corr, "sem_corrs"+save_affix, "./Save_files/")
	# save CPerm results
	# with open (path+"semantic_CPerm"+save_affix+".txt", "w") as outfile:
		# outfile.write("Semantic Variable Result")
		# for index in range(len(sem_result)):
		# 	outfile.write("\n" + sem_titles[index]+":\n")
		# 	if type(sem_result[index]) is list:
		# 		outfile.write("No significant cluster found")
		# 	else:
		# 		sem_result[index].to_string(outfile)
		# outfile.write("\n\n\nLexical Variable Result")
		# for index in range(len(lex_result)):
		# 	outfile.write("\n" + lex_titles[index]+":\n")
		# 	if type(lex_result[index]) is list:
		# 		outfile.write("No significant cluster found")
		# 	else:
		# 		lex_result[index].to_string(outfile)
		# outfile.write("\n\n\nSemantic Variable Partial Correlation Result")
		# for index in range(len(sem_par_result)):
		# 	outfile.write("\n" + sem_par_titles[index]+":\n")
		# 	if type(sem_par_result[index]) is list:
		# 		outfile.write("No significant cluster found")			
		# 	else:
		# 		sem_par_result[index].to_string(outfile)

	return

def load_subjects(mask):
	print ("loading subjects")
	subjects = []
	pbar = tqdm(total = len(EEG_NAME))
	for index in range(len(EEG_NAME)):
		subjects.append(PPData(EEG_NAME[index]))
		subjects[index].load_eeg_from_file(EEG_DIR + EEG_NAME[index], mask = mask)
		subjects[index].clear_artifacts()
		pbar.update(1)
	return subjects

def create_STR(subjects, var_class, IV, CV):
	print ("Creating STR subjects")
	STR_subs = []
	pbar = tqdm(total = len(subjects), file=sys.stdout)
	for subject in subjects:
		sub_name = subject.subname
		sub_eeg = subject.clean_eeg
		sub_bin = subject.bins.reshape(len(subject.bins), 1)
		cond = np.ones((sub_bin.shape[0],1), dtype = str)
		Cond_sti = np.hstack((cond, sub_bin))
		Cond_dict = dict({"1":1})
		sub = var_class.create_STR_subject(sub_name, sub_eeg, IV, extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict, CV = CV)
		STR_subs.append(sub)
		pbar.update(1)
	return np.array(STR_subs)

def create_STR_interpol(subjects, var_class, IV, CV):
	STR_subs = []
	pbar = tqdm(total = len(subjects))
	for subject in subjects:
		sub_name = subject.subname
		sub_eeg = subject.clean_eeg
		sub_bin = subject.bins.reshape(len(subject.bins), 1)
		cond = np.ones((sub_bin.shape[0],1), dtype = str)
		Cond_sti = np.hstack((cond, sub_bin))
		Cond_dict = dict({"1":1})
		sub = var_class.create_STR_subject_tri(sub_name, sub_eeg, IV, extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict, max_diff = 0, decimals = 8, CV = CV)
		STR_subs.append(sub)
		pbar.update(1)
	return np.swapaxes(np.array(STR_subs), 0, 1)

def run_STR(subjects, IV, CV, save_name):
	all_subjects = []
	all_corr_results = np.zeros((len(IV), 256))
	all_subnames = []
	print ("Running STRSA analyses")
	pbar = tqdm(total = (len(subjects)), file=sys.stdout)
	for sub in subjects:
		# print ("#######################  ", end = "")
		# print ("subject " + sub.subname, end = "")
		# print ("  #######################")
		all_subnames.append(sub.subname)
		# sub.print_attribute_info()
		corr_result, var_titles = sub.single_trial_RSA(IV, CV = CV, time_window = 7, step = 1)
		all_subjects.append(corr_result)
		all_corr_results = np.add(all_corr_results, corr_result)
		# print ("")
		pbar.update(1)
	all_corr_results = all_corr_results/len(subjects)
	plot_corr(all_corr_results, np.array(IV), start_end = (-100, 900), interval = 100, save = True, save_name = save_name)
	return np.array(all_subnames), np.array(all_subjects), all_corr_results

def run_STR_interpol(subjects, partial, save_name = "./corr_results.png"):
	all_raw_results = []
	all_results = []
	all_titles = []
	print (subjects.shape)
	for index in range(subjects.shape[1]):
		curr_var = subjects[:, index]
		sub_result = []
		sub_result_alt = []
		for sub in curr_var:
			print ("############################  ", end = "")
			print ("subject " + sub.subname, end = "")
			print ("  ############################")
			sub.print_attribute_info()
			control_var = sub.variable_names[1:]
			if partial == True:
				corr_result, var_titles = sub.single_trial_RSA([sub.variable_names[0]], CV = control_var, time_window = 4, step = 1)
			else:	
				corr_result, var_titles = sub.single_trial_RSA([sub.variable_names[0]], time_window = 6, step = 1)
			sub_result.append(corr_result)
			sub_result_alt.append(corr_result[0,:])
		sub_result = np.array(sub_result)
		all_raw_results.append(np.array(sub_result_alt))
		all_results.append(np.average(sub_result, axis = 0)[0,:])
		all_titles.append(sub.variable_names[0])
	plot_corr(np.array(all_results), ALL_IV, start_end = (-100, 900), interval = 100, save = True, save_name = save_name)
	return np.array(all_raw_results), all_titles

def run_CPerm(all_subjects, all_variables, iteration, save_directory):
	# changing the all_subjects dim from (subject, variable, corr_value) to 
	# (variable, subject, corr_value)
	print ("Running CPerm analyses")
	null_state = np.zeros((all_subjects.shape))
	scale = np.linspace(0, 920, num = 231, dtype = int)
	titles = []
	output_dfs = []
	var_tvals = []	
	var_names = []
	means = []
	stds = []
	for index, var_title in enumerate(all_variables):
		if var_title == "Concreteness":
			var_CPerm, var_matrix, var_pandas_out = CPerm(var_title ,all_subjects[index, :, :], null_state[index, :, :], 10000, show_progress = True, pos_label = scale)
			print (var_title)
			print (var_pandas_out)
			var_tvals.append(var_CPerm.Tvals)
			mean, std = var_CPerm.calculate_std_err()
			var_names.append(var_title)
			means.append(mean)
			stds.append(std)
		# print (var_title)
		# var_CPerm, var_matrix, var_pandas_out = CPerm(var_title ,all_subjects[index, :, :], null_state[index, :, :], iteration, show_progress = True, pos_label = scale)
		# # print (var_title)
		# # print (var_pandas_out)
		# titles.append(var_title)
		# output_dfs.append(var_pandas_out)
		# var_tvals.append(var_CPerm.Tvals)
		# mean, std = var_CPerm.calculate_std_err()
		# var_names.append(var_title)
		# means.append(mean)
		# stds.append(std)
	return titles, output_dfs

def check_senses(words, pos):
	total_sense = 0
	noun_sense = 0
	non_noun_w = 0
	for word in words:
		all_sense = wordnet.synsets(word)
		n_sense = wordnet.synsets(word, pos = pos)
		if len(all_sense) != len(n_sense):
			n_percentage = (len(n_sense)/len(all_sense))*100
			print (word + ": " + str(n_percentage) + "%")
			non_noun_w += 1
		total_sense += len(all_sense)
		noun_sense += len(n_sense)
	all_percentage = (noun_sense / total_sense)*100
	print ("There are " + str(non_noun_w) + " words that has sense other than the specified POS")
	print ("Overall POS sense percentage: " + str(all_percentage) + "%")

def assemble_sim_mat(words):
	results = np.ones((len(words), len(words)))
	pbar = tqdm(total = (len(words)*len(words) - len(words))//2)
	for row_ind in range(0, len(words)):
		for col_ind in range(row_ind+1, len(words)):
			pbar.update(1)
			# results[row_ind, col_ind] = wuSimAvg([words[row_ind]], [words[col_ind]])
			results[row_ind, col_ind] = PathSimDom(words[row_ind], words[col_ind])
			results[col_ind, row_ind] = results[row_ind, col_ind]
	return 1 - results

def wuSimAvg(word1,word2):    
    syns1 = set(ss for word in word1 for ss in wordnet.synsets(word,pos = 'n'))
    syns2 = set(ss for word in word2 for ss in wordnet.synsets(word,pos = 'n'))
    
    sim_value = []
    for syn1 in syns1:
        for syn2 in syns2:
            sim_value.append(wordnet.wup_similarity(syn1,syn2))
    sim_value_mean = np.nanmean(sim_value)
    return sim_value_mean

def wuSimDom(word1, word2):
	syn1 = wordnet.synsets(word1, pos = 'n')[0]
	syn2 = wordnet.synsets(word2, pos = 'n')[0]
	return wordnet.wup_similarity(syn1,syn2)

def PathSimDom(word1, word2):
	syn1 = wordnet.synsets(word1, pos = 'n')[0]
	syn2 = wordnet.synsets(word2, pos = 'n')[0]
	return wordnet.path_similarity(syn1,syn2)

# Other functions
###############################################################################
def save_instance(instance, var_name, directory):
	file_name = directory + var_name + ".pkl"
	with open (file_name, "wb") as output:
		pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)
	return

def load_instance(var_name, directory):
	file_name = directory + var_name + ".pkl"
	with open (file_name, "rb") as input_f:
		instance = pickle.load(input_f)
	return instance

if __name__ == "__main__":
	main()