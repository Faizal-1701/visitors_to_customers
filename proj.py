## importing neccessary libraries

import pandas as pd
import numpy as np
import cv2
import pytesseract # type: ignore
from PIL import Image
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
data = pd.read_csv('D:\\New folder\\Data Science Materials\\Final_Project\\final_dataset_1.csv')
df = pd.read_csv('D:\\New folder\\Data Science Materials\\Final_Project\\final_dataset_1.csv')
data_obj = ['channelGrouping','device_browser','device_operatingSystem','device_deviceCategory','geoNetwork_region','earliest_source','latest_source','earliest_medium','latest_medium']
num = data.select_dtypes(include=['int64','float64','bool']).columns
new_num = ['count_session','count_hit','geoNetwork_region','geoNetwork_latitude','historic_session_page','historic_session','geoNetwork_longitude','last_visitId','sessionQualityDim','single_page_rate','avg_session_time_page','avg_session_time','time_earliest_visit','latest_visit_number','earliest_visit_number','earliest_visit_id','visitId_threshold','latest_visit_id','avg_visit_time','time_latest_visit','bounce_rate','visits_per_day','transactionRevenue','time_on_site','num_interactions','transactionRevenue','time_on_site']
cat = data.select_dtypes(include='object').columns
new_cat = ['channelGrouping','totals_newVisits','device_browser','device_isMobile','device_operatingSystem','device_deviceCategory','latest_medium','earliest_medium','latest_source','latest_isTrueDirect','earliest_isTrueDirect','bounces','has_converted']
data = data.drop('Unnamed: 0', axis=1)
data = data.drop_duplicates()
lb = LabelEncoder()
for i in cat:
    data[i] = lb.fit_transform(data[i])
option = option_menu("Main Menu", ["EDA", "Model Building & Testing", "Image Processing", "NLP", "Recommendation System"], icons=['search', 'brush', 'calculator', 'gear', 'graph-up'], default_index=0,
                        styles={"nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2.5px", "--hover-color":"gray"},
                    "nav-link-selected": {"background-color": "gray"}}, orientation="horizontal")
st.markdown(f""" <style>.stApp {{
                    background:url("https://wallpapers.com/images/high/dull-super-light-black-desktop-sg97fcp36acn0ta3.webp");
                    background-size: cover}}
                    </style>""", unsafe_allow_html=True)
if option == "EDA":
    st.markdown("## :rainbow[Before Data Cleaning]")
    st.write(df.head(),"\n #### :blue[Shape of the dataset] ", data.shape)
    st.markdown("## :rainbow[After Data Cleaning]")
    st.write(data.head(),"\n #### :blue[Shape of the dataset] ", data.shape)
    categorical = data.select_dtypes(include='object').columns
    numerical = data.select_dtypes(include=['int64','float64','bool']).columns
#    st.write(cat, "\n", num)
#    print(data.info())
    sel = st.radio(label= "## :rainbow[Select the input]", options = [':orange[Head]',':red[Tail]',':green[Duplicates]',':yellow[Shape]',':orange[Describe]',':green[Columns]',':red[Null Value]',':yellow[Correlation]'],label_visibility='hidden', horizontal=True, index=7)
    if sel == ':orange[Head]':
        st.write(data.head())
    if sel == ':red[Tail]':
        st.write(data.tail())
    if sel == ':red[Null Value]':
        st.write(data.isnull().sum().reset_index())
    if sel == ':green[Duplicates]':
        st.write(data.duplicated().value_counts())
        st.write(data.shape)
    if sel == ':yellow[Shape]':
        st.write(data.shape)
    if sel == ':orange[Describe]':
        st.write(data.describe())
    if sel == ':green[Columns]':
        st.write(data.columns)
    if sel == ':yellow[Correlation]':
        st.write(data.corr(numeric_only=True))
   
    plots = st.selectbox("Select Your Plot type", options=['Scatter Plot','Relationship Plot','Heat Map','Distribution Plot','Box Plot'])
    x = st.selectbox("Select the x-axis", new_num)
    y = st.selectbox("Select the y-axis", new_num)
    fig, ax = plt.subplots()
    if plots == 'Scatter Plot':
        plt.title('Scatter Plot')
        plt.scatter(x =data[x], y = data[y], color = 'r',label = 'scatter', marker='.')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        st.pyplot(fig)
    if plots == 'Relationship Plot':
        plt.title('Relationship Plot')
        sns.relplot(data, x = data[x], y = data[y], hue = data['device_deviceCategory'], kind= 'scatter')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show() 
        st.pyplot()
    if plots == 'Heat Map':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(25,15))
        sns.heatmap(data.corr(numeric_only=True), vmin = -1, vmax =1,annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        st.pyplot()
    if plots == 'Distribution Plot':
        sns.displot(data,x = x, label = x)
        plt.title("Distribution Plot")
        plt.legend()
        st.pyplot()
    if plots == 'Box PLot':
        plt.boxplot(x)        
        plt.legend()
        st.pyplot(fig)
elif option == "Model Building & Testing":
    st.write(data.head(),"\n",data.shape)
    x = data.drop('has_converted', axis=1)
    y = data[['has_converted']]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =90)    
    def res(pred):
        st.write("Accuracy Score: ",metrics.accuracy_score(y_test, pred))
        st.write("Recall Score: ", metrics.recall_score(y_test, pred))
        st.write("Precision Score: ",metrics.precision_score(y_test, pred))
        st.write("F1 Score: ", metrics.f1_score(y_test, pred))
        st.write("Confusion Matrix: ", metrics.confusion_matrix(y_test, pred))
    model = st.selectbox("SELECT YOUR MODEL", options = ['Logistic Regression','Decision Tree Classifier','KNN', 'SVC'])
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
elif option == "Image Processing":
    st.set_option('deprecation.showfileUploaderEncoding', False) 
    def perform_ocr(image):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Perform OCR
        ocr_result = pytesseract.image_to_string(gray_image)
        return ocr_result
    img = st.file_uploader('give me a image', type=['png','jpg','jpeg'],label_visibility='hidden',accept_multiple_files=False)
    if img is not None:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8) # Convert the file object to a numpy array
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the numpy array to an image
        ocr_result = perform_ocr(img)
        if ocr_result:
            # Display OCR result
            st.write("OCR Result:")
            st.write(ocr_result)
        else:
            st.write("No text found in the uploaded image.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(":green[Normal Image]")
            st.image(img, caption= 'Normal Image')
            st.write(":green[Grayscale Image]")
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # image to grayscale            
            # Display the grayscale image using OpenCV (not recommended for Streamlit)
            st.image( gray_img,caption = 'Grayscale Image')

        with col2:
            st.write(":green[Blur Image]")
            blur = cv2.GaussianBlur(gray_img, (11,11),0)
            st.image(blur, caption='Blur Image')
            st.write(":green[Edge Detect Image]")
            edges = cv2.Canny(blur, 50,150)
            st.image(edges, caption = "Edge Detect Image")

        with col3:
            st.write(":green[De-Blur Image Image]")
            psf = np.ones((5, 5)) / 25  # Example Gaussian PSF
            deblurred_img = cv2.filter2D(blur, -1, psf)
            st.image(deblurred_img,caption='De-Blur Image')

            # Define a sharpening kernel
            st.write(":green[Sharp Image]")
            kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

            # Apply the kernel to perform sharpening
            sharpened = cv2.filter2D(img, -1, kernel)
            st.image(sharpened, caption='Sharp Image')







# import pandas as pd

# # Your DataFrame
# a = [1,2,3,4,5,6,7,8,9,10]
# b = [9,8,7,4,5,6,2,1,3,0]
# c = [5,4,7,8,1,2,4,5,6,4]
dat = pd.DataFrame(data, columns=data.columns)

# Drop duplicates in each column and count remaining values
remaining_values = dat.apply(lambda x: len(x) - x.drop_duplicates().shape[0])

# Print remaining values
st.write("Remaining values after dropping duplicates in each column:")
st.write(remaining_values)
