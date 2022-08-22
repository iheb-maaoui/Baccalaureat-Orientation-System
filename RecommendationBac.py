import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd

st.title("Recommmandation Orientation Bac")
st.write("\n")
st.write("\n")

col1, col2, col3  = st.beta_columns(3)
coll1, coll2, coll3, coll4, coll5  = st.beta_columns(5)

@st.cache(allow_output_mutation=True)
def loadData():
	df = pd.read_csv("P2MUpdated.csv")
	
	del(df['Horodateur'])
	del(df['Est-ce tu as choisi les math comme option?'])
	return df


def questions(df,xuser):
	sections=['Lettres & langues ','Sc. Humaines','Médias','Sc. Juridiques','Arts et métiers','Médicale','Gestion et sc. Économiques','Sc. Fondamentales','Agriculture & environnement','Etudes ingénieurs OU Etudes technologiques']




	s={'Lettres & langues ':"""Institut Préparatoire aux Etudes Littéraires et de Sciences Humaines de Tunis \n""",'Sc. Humaines': """Faculté des Sciences Humaines et Sociales de Tunis \n
	Institut Supérieur des Etudes Appliquées en Humanités de Tunis \n
	Institut Supérieur des Études Appliquées en Humanités de Zaghouan\n """

	,'Médias':"""Institut de presse et des sciences de l'information\n
	Université centrale
	"""

	,'Sc. Juridiques': """ Faculté des Sciences Juridiques, Politiques et Sociales de Tunis\n
	Sciences Po Tunis : Institut d’études politiques de Tunis (IEP Tunis)\n
	Faculté de Droit et des Sciences Politiques de Tunis\n""" 



	,'Arts et métiers': """ Tunis Institut Supérieur des Beaux Arts de Tunis\n
	Tunis Institut Supérieur de Musique\n
	Tunis Institut Supérieur d'Art Dramatique\n"""

	,'Médicale':"""Faculté de Médecine de Monastir\n
	Faculté de Médecine de Sfax\n
	Faculté de Médecine de Sousse\n
	Faculté de Médecine de Tunis\n
	École Supérieure des Sciences et Techniques de la Santé de Tunis\n"""

	,'Gestion et sc. Économiques':"""Ecole Supérieure des Sciences Economiques et Commerciales de Tunis\n
	Institut Supérieur de Gestion de Tunis\n
	Faculté des Sciences Economiques et de Gestion de Nabeul\n
	Institut des Hautes Etudes Commerciales de Carthage\n
	FSEG Tunis El Manar\n """

	,'Sc. Fondamentales':"""Faculté des sciences\n
	École Normale Supérieur de Tunis\n"""



	,'Agriculture & environnement':"""Ecole Supérieure d'Agriculture de Mateur\n
	Ecole supérieure de l'agriculture de Mograne\n
	Institut National Agronomique de Tunisie\n"""

	,'Etudes ingénieurs OU Etudes technologiques':"""Institut National des Sciences Appliquées et de Technologie\n
	Institut Préparatoire aux Etudes d'Ingénieurs Tunis \n
	Institut Préparatoire aux Etudes d'Ingénieur Manar\n
	Institut Préparatoire aux Etudes d'Ingénieur Monastir \n
	Institut Préparatoire aux Etudes d'Ingénieur Sousse \n
	ISET\n
	ESPRIT Prepa\n"""








		}

	d={'Sc. Humaines':"""Sous l’appellation “Sciences humaines et sociales (SHS)”, on désigne un ensemble de disciplines consacrées à divers aspects de la réalité humaine, examinant le plan de l'individu et le plan collectif. Philosophie, histoire, géographie, sociologie, psychologie, linguistique… Les domaines de recherche des sciences humaines et sociales sont multiples et variés !""",'Lettres & langues ': """ Les Lettres sont destinées à ceux qui s'intéressent à la littérature, aux langues, à l'histoire-géo… et qui sont curieux des différentes formes de culture littéraire fondée sur l'analyse, la mise en perspective d'une œuvre et l'argumentation. """

	,'Médias':""" études en médias permettent d’acquérir des connaissances sur les techniques de communication, marketing et management liés aux médias pour nous informer et nous divertir.
	"""

	,'Sc. Juridiques': """ la formation en « Sciences juridiques » a pour objectifs de :
	- Dispenser les notions fondamentales, les fondements théoriques et méthodologiques en Droit et en Sciences Politiques permettant aux étudiants de comprendre, de maîtriser les dimensions pratiques du droit et d’analyser le fonctionnement des institutions Nationales et Internationales.""" 



	,'Arts et métiers': """ Cette classe de programmes d’enseignement comprend tout programme général conçu pour permettre aux apprenants de s’initier aux arts visuels et d’acquérir des connaissances dans ce domaine. Ces programmes comprennent des cours portant sur l’art, la photographie et d’autres moyens de communication visuels."""

	,'Médicale':""" Professionnel central du monde de la santé, le médecin examine et établit un diagnostic, ordonne des examens et prescrit le traitement pharmaceutique ou hospitalier qui s'impose. Le médecin suit ses malades et gère leur santé dans sa globalité. Il y a des médecins généralistes et des médecins spécialistes d'une pathologie."""

	,'Gestion et sc. Économiques':""" Réputée exigeante en mathématiques, cette licence n’en reste pas moins pluridisciplinaire et propose même au moins matheux des étudiants une formation abordable dans le domaine de l’économie."""

	,'Sc. Fondamentales':""" Sciences de la vie, physique, informatique, mécanique, génie civil... l'université propose des licences dans tous les domaines scientifiques. Les formations associent théorie et mises en application. Elles ouvrent de nombreuses perspectives, notamment en master ou en école d'ingénieurs."""



	,'Agriculture & environnement':""" L’agronomie correspond à la science de l’agriculture. Il s’agit d’un secteur qui met en lien les productions agricoles, la protection et l’adaptation à l’environnement, ainsi que l’industrie agroalimentaire qui transforme ensuite les matières premières qu’elles soient d’origine végétale ou issues de la production animale."""

	,'Etudes ingénieurs OU Etudes technologiques':""" Un ingénieur peut être défini comme quelqu’un qui, confronté à des problèmes d’ordre technique, tente d’y apporter des solutions elles-mêmes techniques. Il intervient à différents niveaux, de création, de conception, de réalisation, de mise en œuvre et de contrôle de produits, de systèmes ou de services ."""








		}
		 
		 
	
def catToNum(df,variable):
	classes = [1,0,2]
	RealClasses = list(df[variable].unique())
	RealClasses.sort()
	
	for i in range(3):
		df[variable] = np.where(df[variable]==RealClasses[i],classes[i],df[variable])

def null_cleaning(df):
	for i in df.columns:
			df[i] = np.where(df[i].isnull(),df[i].value_counts().index[[df[i].value_counts().argmax()]][0],df[i])

def catToNum2(df,variable,l):
	## Pas du tout -----> 0
	## Moyennement ------> 1
	## Oui Parfaitement ou Beaucoup------> 2
	classes = [0,1,2]
	RealClasses = ['Pas du tout', 'Moyennement', l]
	for i in range(3):
		df[variable] = np.where(df[variable]==RealClasses[i],classes[i],df[variable])

#@st.cache
def preprocessing(df):
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Lettres & langues',0,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Sc. Humaines',1,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Médias',2,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Sc. Juridiques',3,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Arts et métiers',4,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Médicale',5,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Gestion et sc. Économiques',6,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Sc. Fondamentales',7,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Agriculture & environnement',8,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Etudes techno',9,df['Quel est ton parcours académique ?'])
	df['Quel est ton parcours académique ?'] = np.where(df['Quel est ton parcours académique ?'] == 'Etudes ingénieurs & architech',9,df['Quel est ton parcours académique ?'])

	variable = 'Combiens de livres as tu  lu les 6 derniers mois?'
	catToNum(df,variable)

	catToNum(df,'Combiens de pages as tu écrites?')

	variable = 'As tu fait des tentatives de poésie?'
	df[variable] = np.where(df[variable]=='Non','<2',df[variable])
	catToNum(df,'As tu fait des tentatives de poésie?')

	variable = "Lis tu  des livres d'histoire?"
	df[variable] = np.where(df[variable]=='Non','<3',df[variable])

	catToNum(df,"Lis tu  des livres d'histoire?")

	L2 = []
	L1 = []
	RealClasses = ['Pas du tout', 'Moyennement', 'Oui parfaitement']
	for i in df.columns:
			aux = df[i].unique()
			if 'Pas du tout' in aux:
					if 'Beaucoup' in aux:
						L1.append(i)
					else:
						L2.append(i)
	if(len(L1)!=0):
		for variable in L1:
				catToNum2(df,variable,'Beaucoup')

	if(len(L2)!=0):
		for variable in L2:
				catToNum2(df,variable,'Oui parfaitement')

	variable = 'Combiens de journaux par semaine tu lis?'
	catToNum(df,variable)

	variable = "Dans le présent gouvernement: combien de ministres peux tu identifier?"
	df[variable] = np.where(df[variable]=='Tous',2,df[variable])
	df[variable] = np.where(df[variable]=='<5',0,df[variable])
	df[variable] = np.where(df[variable]=='>5',1,df[variable])

	variable ='Votre moyenne des sciences naturelles en 3eme annee secondaire'
	df[variable] = np.where(df[variable]=='>15',2,df[variable])
	df[variable] = np.where(df[variable]=='<10',0,df[variable])
	df[variable] = np.where(df[variable]=='10 à 15',1,df[variable])

	variable = 'Est-ce tu veux te consacrer aux autres?'
	df[variable] = np.where(df[variable]=='Un peu',1,df[variable])
	df[variable] = np.where(df[variable]=='Oui souvent',2,df[variable])
	

	import category_encoders as ce
	#Create object for one-hot encoding
	L= ['As tu  visité un monument historique?',"Comment exprimes tu  tes idées et tes émotions?",'Joues tu le scrable?','Votre Réaction face à une nouvelle?',"Est-ce que  tu joues à l'echec?",'Devant une décision:',"Lors d'un discours:","Lors d'une discussion?",'Dans ton travail, tu préfères:']#'Est-ce tu as choisi les math comme option?']  
	encoder=ce.OneHotEncoder(cols=L,handle_unknown='return_nan',return_df=True,use_cat_names=True)
	data_encoded = encoder.fit_transform(df)


	correlation_matrix=data_encoded.corr()
	correlated_features=[]
	for i in range(len(correlation_matrix .columns)):
			for j in range(i):
				if abs(correlation_matrix.iloc[i, j]) >= 0.73:
						colname = correlation_matrix.columns[i]
						correlated_features.append(colname)

	data_encoded.drop(correlated_features, axis='columns', inplace=True)

	return data_encoded


def LR(Xtrain_fs,ytrain,Xtest_fs):
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import KFold
	lr = LogisticRegression(random_state=1,max_iter=1000)
	lr.fit(Xtrain_fs, ytrain)
	ypred = lr.predict(Xtest_fs)
	return ypred

def SVM(Xtrain_fs,ytrain,Xtest_fs):
	from sklearn import svm
	clf = svm.SVC(kernel='linear') # Linear Kernel
	clf.fit(Xtrain_fs, ytrain)
	ypred = clf.predict(Xtest_fs)
	return ypred

def MLP(Xtrain_fs,ytrain,Xtest_fs):
	from sklearn.neural_network import MLPClassifier
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,), random_state=1)
	clf.fit(Xtrain_fs, ytrain)
	ypred = clf.predict(Xtest_fs)
	return ypred

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def select_features(Xtrain, ytrain, Xtest,i):
	fs = SelectKBest(score_func=chi2, k=i)
	fs.fit(Xtrain, ytrain)
	Xtrain_fs = fs.transform(Xtrain)
	Xtest_fs = fs.transform(Xtest)
	return Xtrain_fs, Xtest_fs, fs

def final_model(Xtrain, ytrain, Xtest):
	Xtrain_fs, Xtest_fs, fs = select_features(Xtrain, ytrain, Xtest,'all')
	y1 = np.array(LR(Xtrain_fs,ytrain,Xtest_fs))
	y2 = np.array(SVM(Xtrain_fs,ytrain,Xtest_fs))
	y3 = np.array(MLP(Xtrain_fs,ytrain,Xtest_fs))
	
	ypred = np.zeros(y1.shape)
	ypred = np.where(y1==y2,y1,y3)
	ypred = np.where(ypred==y2,ypred,y3)
	ypred = np.where(ypred==y1,ypred,y3)
	
	return ypred

def balanceDS(data_encoded):

	X = data_encoded.iloc[:,1:].values
	y = data_encoded.iloc[:,0].values
	X=X.astype('int')
	y=y.astype('int')

	from imblearn.over_sampling import RandomOverSampler
	ros = RandomOverSampler(random_state=0)
	X, y = ros.fit_resample(X, y)

	
	
	return X,y

def main():
	k=2
	df=loadData()
	null_cleaning(df)
	xuser = pd.DataFrame(columns = df.columns)
	xuser['Quel est ton parcours académique ?']=pd.Series([8])


	with col1:
		pass
	with col3:
		pass
	with col2 :
		submit=st.button("--Confirmer--")

	xuser['Combiens de livres as tu  lu les 6 derniers mois?']=st.sidebar.selectbox('Combiens de livres as tu lu les 6 derniers mois?',
				['2 à 4', '>4', '<2'])


	xuser['Comment exprimes tu  tes idées et tes émotions?']=st.sidebar.selectbox('Comment exprimes tu tes idées et tes émotions?',
				['Réaction verbale', 'Ecrire', 'Réaction physique'])

	xuser['Combiens de pages as tu écrites?']=st.sidebar.selectbox('Combiens de pages as tu écrites?',
				['<5', '>10', '5 à 10'])


	xuser['As tu fait des tentatives de poésie?']=st.sidebar.selectbox('As tu fait des tentatives de poésie?',
				['>2', '1 à 2', 'Non'])
	xuser['Joues tu le scrable?']=st.sidebar.selectbox('Joues tu le scrable?',
				['Non', 'Parfois', 'Souvent'])

	xuser['As tu  visité un monument historique?']=st.sidebar.selectbox('As tu visité un monument historique?',
				['Oui', 'Non'])

	xuser["Lis tu  des livres d'histoire?"]=st.sidebar.selectbox('Lis tu des livres d histoire?',
				['>3', '1 à 3', 'Non'])

	xuser['Votre Réaction face à une nouvelle?']=st.sidebar.selectbox('Votre Réaction face à une nouvelle?',
				['Admettre','Mettre en question', 'Négliger'])

	xuser['Combiens de journaux par semaine tu lis?']=st.sidebar.selectbox('Combiens de journaux par semaine tu lis?',
				['<3', '3 à 7', '>7'])

	xuser['Dans le présent gouvernement: combien de ministres peux tu identifier?']=st.sidebar.selectbox('Dans le présent gouvernement: combien de ministres peux tu identifier?',
				['<5' ,'>5', 'Tous'])

	xuser['Devant une décision:']=st.sidebar.selectbox('Devant une décision:',
				['Tu prends la décision tout seul indépendamment des autres',
	'Discuter ton point de vue, puis décider',
	'Laisser les autres décider à ta place'])

	xuser["Lors d'un discours:"]=st.sidebar.selectbox('Lors d un discours:',
				['Tu sais séduire les autres',
	"Tu n'aimes pas te dévoiler devant les autres",
	"Tu t'exprimes mais timidement"])

	xuser["Lors d'une discussion?"]=st.sidebar.selectbox('Lors d une discussion?',
				['Tu participes et tu acceptes changer tes idées?',
	'Tu ne participes pas et tu préfères garder tes idées à toi?',
	'Tu participes et tu tiens à tes idées?'])
	xuser['Est-ce tu veux te consacrer aux autres?']=st.sidebar.selectbox('Est-ce tu veux te consacrer aux autres?',['Pas du tout', 'Un peu', 'Oui souvent'])

	xuser["Est-ce que  tu joues à l'echec?"]=st.sidebar.selectbox('Est-ce que tu joues à l echec?',
				['Non' ,'Rarement' ,'Oui souvent'])


	xuser['Dans ton travail, tu préfères:']=st.sidebar.selectbox('Dans ton travail, tu préfères:',
				["L'essentiel un travail peu fatiguant", 'Un tarvail de bureau',
	'Un travail actif'])
	xuser['Dans ton travail, tu préfères, une activité artistique']=st.sidebar.selectbox('Dans ton travail, tu préfères, une activité artistique',
				['Moyennement', 'Pas du tout', 'Beaucoup']
	)

	xuser['Dans ton travail, tu aimes créer, inventer …']=st.sidebar.selectbox('Dans ton travail, tu aimes créer, inventer …',
				['Pas du tout', 'Moyennement', 'Beaucoup'])

	xuser['En cas de stress, je garde la tête froide']=st.sidebar.selectbox('En cas de stress, je garde la tête froide',
				['Moyennement', 'Oui parfaitement', 'Pas du tout'])

	xuser['Je repère facilement les petits défauts']=st.sidebar.selectbox('Je repère facilement les petits défauts',
				['Moyennement', 'Oui parfaitement', 'Pas du tout'])

	xuser['Tu es du genre à répondre aux besoins des autres']=st.sidebar.selectbox('Tu es du genre à répondre aux besoins des autres',
				['Oui parfaitement', 'Moyennement', 'Pas du tout'])

	xuser['Votre moyenne des sciences naturelles en 3eme annee secondaire']=st.sidebar.selectbox('Votre moyenne des sciences naturelles en 3eme annee secondaire',
				['<10' ,'10 à 15', '>15'])

	xuser["Je me sens à l'aise en travaillant en équipe"]=st.sidebar.selectbox('Je me sens à l aise en travaillant en équipe',
				['Pas du tout', 'Moyennement' ,'Oui parfaitement'])

	xuser['Je trouve du plaisir à travailler les mathématiques']=st.sidebar.selectbox('Je trouve du plaisir à travailler les mathématiques',
				['Moyennement' ,'Pas du tout', 'Oui parfaitement'])

	xuser["Ce qu'il t'attire le plus, les sciences fondamentales (math, physiques…)."]=st.sidebar.selectbox('Ce qu il t attire le plus, les sciences fondamentales (math, physiques…).',
				['Moyennement' ,'Pas du tout' , 'Oui parfaitement'])


	xuser["Ce qu'il t'attire le plus, transmettre tes connaissances aux autres."]=st.sidebar.selectbox('Ce qu il t attire le plus, transmettre tes connaissances aux autres.',
				['Moyennement' ,'Oui parfaitement', 'Pas du tout'])



	xuser["Ce qu'il t'attire le plus, travailler avec tes mains."]=st.sidebar.selectbox('Ce qu il t attire le plus, travailler avec tes mains',
				['Pas du tout', 'Moyennement', 'Oui parfaitement'])

	xuser["Ce qu'il t'attire le plus, utiliser des outils ou des instruments."]=st.sidebar.selectbox('Ce qu il t attire le plus, utiliser des outils ou des instruments.',
				['Pas du tout' ,'Oui parfaitement', 'Moyennement'])



	xuser["Ce qu'il t'attire le plus, mettre en application tes connaissances théoriques."]=st.sidebar.selectbox("Ce qu'il t'attire le plus, mettre en application tes connaissances théoriques.",
				['Pas du tout', 'Moyennement', 'Oui parfaitement'])



	xuser["Ce qu'il t'attire le plus, réaliser des projets."]=st.sidebar.selectbox("Ce qu'il t'attire le plus, réaliser des projets.",
				['Moyennement' ,'Pas du tout', 'Oui parfaitement'])

	xuser['Pour toi, le travail en équipe, est une obligation']=st.sidebar.selectbox('Pour toi, le travail en équipe, est une obligation',
				['Moyennement', 'Pas du tout' ,'Oui parfaitement'])




	xuser["Ce qu'il t'attire le plus, faire preuve d'endurance."]=st.sidebar.selectbox("Ce qu'il t'attire le plus, faire preuve d'endurance.",
				['Beaucoup', 'Moyennement' ,'Pas du tout'])



	xuser["Tu est du type qui aime travailler la terre."]=st.sidebar.selectbox("Tu est du type qui aime travailler la terre.",
				['Pas du tout','Moyennement','Beaucoup'])



	xuser["Tu est du type qui aime s'occuper des animaux."]=st.sidebar.selectbox("Tu est du type qui aime s'occuper des animaux.",
				['Moyennement', 'Pas du tout', 'Beaucoup'])
	
	
	df=df.append(xuser , ignore_index=True)
	data_encoded = preprocessing(df)
	xuser_encoded = data_encoded.loc[len(df)-1]
	data_encoded.drop(len(df)-1,0,inplace=True)
	del(xuser_encoded['Quel est ton parcours académique ?'])
	X,y=balanceDS(data_encoded)
	ypred = final_model(X,y,xuser_encoded.values.reshape(1, -1))
	st.write("\n")
	st.write("\n")
	st.write('\n')
	st.header("*On recommande :*")
	st.subheader(sections[ypred[0]]+" :")
	st.write('\n')
	st.write(d[sections[ypred[0]]])
	st.write('\n')
	st.header("*Vous pouvez choisir parmi ces écoles :*")
	st.write('\n')
	st.write(s[sections[ypred[0]]])
	if submit:
		questions(df,xuser)

	  

	 

	
