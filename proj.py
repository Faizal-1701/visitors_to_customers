## importing neccessary libraries
import pandas as pd
import numpy as np
import cv2, re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn import metrics, tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

## streamlit parts
st.set_page_config(page_title= 'Visiter to customer',layout='wide', page_icon='chart_with_upwards_trend', initial_sidebar_state='expanded')
option = option_menu("Main Menu", ["EDA", "Model Building & Testing", "Image Processing", "NLP", "Recommendation System"], icons=['search', 'brush', 'calculator', 'gear', 'graph-up'], default_index=0,
                    styles={"nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2.5px", "--hover-color":"gray"},
                    "nav-link-selected": {"background-color": "gray"}}, orientation="horizontal")
st.markdown(f""" <style>.stApp {{
                    background:url("https://wallpapers.com/images/high/dull-super-light-black-desktop-sg97fcp36acn0ta3.webp");
                    background-size: cover}}
                    </style>""", unsafe_allow_html=True)
## data access
data = pd.read_csv('D:\\New folder\\Data Science Materials\\Final_Project\\final_dataset_1.csv')
df = pd.read_csv('D:\\New folder\\Data Science Materials\\Final_Project\\final_dataset_1.csv')
new_num = ['count_session','count_hit','geoNetwork_region','geoNetwork_latitude','historic_session_page','historic_session','geoNetwork_longitude','last_visitId','sessionQualityDim','single_page_rate','avg_session_time_page','avg_session_time','time_earliest_visit','latest_visit_number','earliest_visit_number','earliest_visit_id','visitId_threshold','latest_visit_id','avg_visit_time','time_latest_visit','bounce_rate','visits_per_day','transactionRevenue','time_on_site','num_interactions','transactionRevenue','time_on_site']
cat = data.select_dtypes(include='object').columns
data = data.drop('Unnamed: 0', axis=1)
data = data.drop_duplicates()
lb = LabelEncoder() ## encoding
for i in cat:
    data[i] = lb.fit_transform(data[i])
if option == "EDA":
    st.markdown("## :rainbow[Before Data Cleaning]")
    st.write(df.head(),"\n #### :blue[Shape of the dataset] ", df.shape)
    st.markdown("## :rainbow[After Data Cleaning]")
    st.write(data.head(),"\n #### :blue[Shape of the dataset] ", data.shape)
    #categorical = data.select_dtypes(include='object').columns
    #numerical = data.select_dtypes(include=['int64','float64','bool']).columns
    sel = st.radio(label= "## :rainbow[Select the input]", options = [':orange[Head]',':red[Tail]',':green[Duplicates]',':yellow[Shape]',':orange[Describe]',':green[Columns]',':red[Null Value]',':yellow[Correlation]'],label_visibility='visible', horizontal=True, index=7)
    if sel == ':orange[Head]':
        st.write(data.head())
    if sel == ':red[Tail]':
        st.write(data.tail())
    if sel == ':red[Null Value]':
        st.write(data.isnull().sum().reset_index())
    if sel == ':green[Duplicates]':
        st.write(data.duplicated().value_counts())
        st.write(df.shape)
    if sel == ':yellow[Shape]':
        st.write(data.shape)
    if sel == ':orange[Describe]':
        st.write(data.describe())
    if sel == ':green[Columns]':
        st.write(data.columns)
    if sel == ':yellow[Correlation]':
        st.write(data.corr(numeric_only=True))
    plots = st.selectbox("## :red[Select Your Plot type]", options=['Hist Plot','Relationship Plot','Heat Map','Distribution Plot'])
    x = st.selectbox("# :violet[Select the x-axis]", new_num)
    y = st.selectbox("# :violet[Select the y-axis]", new_num)
    fig, ax = plt.subplots()
    if plots == 'Hist Plot':
        plt.title('Hist Plot')
        sns.histplot(data= data,x =data[x] ,color = 'r',bins = 20,element='bars', kde=True, linewidth = 1.2, edgecolor ='black')
        plt.tight_layout()
        plt.grid(True, linestyle = "--")
        plt.xlabel(x)
        st.pyplot(fig)
    if plots == 'Relationship Plot':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.title('Relationship Plot')
        sns.relplot(data, x = data[x], y = data['has_converted'], kind= 'scatter')
        plt.xlabel(x)
        plt.ylabel('Counts')
        st.pyplot()
    if plots == 'Heat Map':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(25,15))
        sns.heatmap(data.corr(numeric_only=True), vmin = -1, vmax =1,annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        st.pyplot()
    if plots == 'Distribution Plot':
        sns.displot(data, label = x)
        plt.title("Distribution Plot")
        plt.legend()
        st.pyplot()
## model evaluation & testing
elif option == "Model Building & Testing":
    st.write(data.head(),"\n",data.shape)
    x = data.drop('has_converted', axis=1)
    y = data[['has_converted']]  # target
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =90)    
    def res(pred):
        st.write(":orange[Accuracy Score: ]",metrics.accuracy_score(y_test, pred))
        st.write(":red[Recall Score: ]", metrics.recall_score(y_test, pred))
        st.write(":green[Precision Score:] ",metrics.precision_score(y_test, pred))
        st.write(":violet[F1 Score: ]", metrics.f1_score(y_test, pred))
        st.write(":rainbow[Confusion Matrix:] ", metrics.confusion_matrix(y_test, pred))
    model = st.selectbox(":rainbow[SELECT YOUR MODEL]", options = ['Logistic Regression','Decision Tree Classifier','KNN', 'SVC'])
    if model == 'Logistic Regression':
        lr = LogisticRegression()
        lr_model = lr.fit(x_train,y_train)
        ypred = lr_model.predict(x_test)
        res(ypred)
    elif model == 'Decision Tree Classifier':
        dtc = DecisionTreeClassifier()
        dtc_model = dtc.fit(x_train, y_train)
        ypred = dtc_model.predict(x_test)
        res(ypred)
#        tree.export_graphviz(dtc, out_file='tree.dot',class_names=data['has_converted'])
    elif model == 'KNN':
        knn = KNeighborsClassifier()
        knn_model = knn.fit(x_train, y_train)
        ypred = knn_model.predict(x_test)
        res(ypred)
    elif model == 'SVC':
        svc = SVC().fit(x_train, y_train)
        ypred = svc.predict(x_test)
        res(ypred)
## image processing
elif option == "Image Processing":
   st.set_option('deprecation.showfileUploaderEncoding', True) 
   img = st.file_uploader('#### :rainbow[Give me a image]', type=['png','jpg','jpeg'],label_visibility='visible',accept_multiple_files=False)
   if img is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(":green[Normal Image]")
            st.image(img, caption= 'Normal Image')
            st.write(":green[Grayscale Image]")
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8) # Convert the file object to a numpy array
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the numpy array to an image
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # image to grayscale            
            st.image( gray_img,caption = 'Grayscale Image')
        with col2:
            st.write(":green[Blur Image]")
            blur = cv2.GaussianBlur(gray_img, (11,11),0)
            st.image(blur, caption='Blur Image')
            st.write(":green[Edge Detect Image]")
            edges = cv2.Canny(blur, 50,150)
            st.image(edges, caption = "Edge Detect Image")
        with col3:
            st.write(":green[De-Blur Image]")
            psf = np.ones((5, 5)) / 25  # Example Gaussian PSF
            deblurred_img = cv2.filter2D(blur, -1, psf)
            st.image(deblurred_img,caption='De-Blur Image')
            st.write(":green[Sharp Image]")
            kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
            sharpened = cv2.filter2D(img, -1, kernel)
            st.image(sharpened, caption='Sharp Image')
        # reader = easyocr.Reader(['en'])
        # result = reader.readtext(img, detail = 0)
        # st.write(result)

elif option == 'NLP':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    import nltk
    from nltk import ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist
    from wordcloud import WordCloud 
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('words')
    nltk.download('vader_lexicon') 
    nltk.download('maxent_ne_chunker')
    txt = st.text_input("#### :green[Enter U R Text Here....]")
    st.write(txt)  # print the text
    
    ## cleaning text
    cln_anl = st.selectbox('##### :violet[Select the category]',['Cleaning Parts', 'Analyze Parts'])
    if cln_anl == 'Cleaning Parts':
        opt = st.radio("##### :red[Please, select the option]", options=[':blue[Stopwords]',':green[Stemming]', ':orange[Word_tokens]',':yellow[Sent_tokens]'], horizontal=True)
        txt = re.sub(r'\W+', ' ', txt).lower() # remove non-alphabet 
        txt = txt.replace("[^a-zA-Z]", " ")
        if opt == ':orange[Word_tokens]':
            word_tokens = word_tokenize(txt) 
            st.write(word_tokens) 
        if opt ==':yellow[Sent_tokens]':
            sent_tokens = sent_tokenize(txt)
            st.write(sent_tokens)
        if opt ==':green[Stemming]':
            ps = PorterStemmer()
            stem = [ps.stem(word) for word in word_tokenize(txt)]
    #        st.write(" ".join(stem))
            st.write(stem)
        ## stop words
        if opt == ':blue[Stopwords]':
            words = word_tokenize(txt)
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            st.write(filtered_words)
       
    if cln_anl == 'Analyze Parts':
        opt = st.radio('##### :red[Please, select the option]',options=[':red[POS]',':blue[NER]',':green[Keyword Extract]',':yellow[WordCloud]',':orange[Sentiment Analysis]'], horizontal=True)
        if opt ==':red[POS]':
            sent_2 = word_tokenize("Barack Obama was born in Hawaii. He was the 44th President of the United States.")
            token = []
            for tok in sent_2:
                token.append(nltk.pos_tag([tok]))
            st.write(token)
        if opt ==':blue[NER]':
            sent_2 = word_tokenize("Barack Obama was born in Hawaii. He was the 44th President of the United States.")
            named_entities = []
            ne_chunked = ne_chunk(nltk.pos_tag(sent_2))
            for chunk in ne_chunked:
                if hasattr(chunk, 'label'):
                    entity = ' '.join(c[0] for c in chunk)
                    entity_type = chunk.label()
                    named_entities.append((entity, entity_type))
            # Print named entities
            st.write(":rainbow[Named Entities: ] ")
            for entity, entity_type in named_entities:
                st.write(f"{entity} - {entity_type}")
         ## KEyword Extract
        if opt == ':green[Keyword Extract]':
            sentences = sent_tokenize(txt)
            words = word_tokenize(txt)
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            filtered_words = [''.join(char for char in word if char.isalpha()) for word in filtered_words] # remove non-alphabets
            word_frequencies = FreqDist(filtered_words)  
#            st.write(word_frequencies)   # Calculate word frequencies
            sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True) # order the word freq
            top_keywords = [word for word, freq in sorted_word_frequencies[:10]]  # Adjust the number as needed
            st.write(":rainbow[Top Keywords : ]", len(top_keywords))
            for keyword in top_keywords:
                st.write(keyword)

        if opt ==':yellow[WordCloud]':
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(txt)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot()

        if opt== ':orange[Sentiment Analysis]':
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = sid.polarity_scores(txt)
            st.write(":rainbow[Sentiment Scores:]", sentiment_scores)
            if sentiment_scores['compound'] >= 0.05:
                st.write(":rainbow[Overall Sentiment: Positive]")
            elif sentiment_scores['compound'] <= -0.05:
                st.write(":rainbow[Overall Sentiment: Negative]")
            else:
                st.write(":rainbow[Overall Sentiment: Neutral]")

if option == "Recommendation System":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    user_input = st.text_input("##### :green[Enter the name of the product: ]")
    data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Camera'],
        'description': ['Powerful laptop with high-performance specs', 
                        'Latest smartphone with advanced features', 
                        'Premium headphones for immersive audio experience', 
                        'Compact tablet for on-the-go productivity', 
                        'High-resolution camera for professional photography']}
#    data = pd.read_csv("E:/Project/ratings.csv")
 #   data.head()
    if st.button("Get Recommendation"):
        df = pd.DataFrame(data)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['description'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)  # Compute cosine similarity matrix
        try:
            def get_recommendations(product_name, cosine_sim=cosine_sim):
                idx = df[df['product_name'] == product_name].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:6]
                product_indices = [i[0] for i in sim_scores]
                return df['product_name'].iloc[product_indices]
            recommendations = get_recommendations(user_input)
            st.write(":rainbow[Recommended Products: ]", recommendations)
        except :
            st.write(":red[Please enter the word properly]")

