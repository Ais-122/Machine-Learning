import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



st.title("PENAMBANGAN DATA B")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("##### Nama  : Aisyiyah Maulana Wibawati ")
    st.write("##### Nim   : 200411100122 ")
    st.write("##### Kelas : Penambangan Data B ")
    st.write("""# Data Set Description """)
    st.write("###### Data set ini Adalah : Classification on Indian Liver Patient (Klasifikasi Pasien Liver India) ")
    st.write("###### Sumber Data Set dari Kaggle : https://raw.githubusercontent.com/Ais-122/Machine-Learning/main/indian_liver_patient.csv")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. Age (Umur ) : 
    Umur atau usia pada manusia adalah waktu yang terlewat sejak kelahiran. Semisal, umur manusia dikatakan lima belas tahun diukur sejak dia lahir hingga waktu umur itu dihitung.""")
    st.write("""2. Gender (Jenis Kelamin) : 
    Gender atau jantina adalah serangkaian karakteristik yang terikat kepada dan membedakan maskulinitas dan femininitas. Karakteristik tersebut dapat mencakup jenis kelamin, hal yang ditentukan berdasarkan jenis kelamin, atau identitas gender.
    """)
    st.write("""3. Total_Bilirubin (Bilirubin Total) :
    Cek bilirubin total adalah suatu pemeriksaan yang dilakukan untuk mengukur jumlah total bilirubin yang ada di dalam darah. Tes ini bertujuan untuk mengevaluasi fungsi hati atau membantu mendiagnosis anemia yang disebabkan oleh kerusakan sel darah merah (anemia hemolitik).
    """)
    st.write("""4. Direct_Bilirubin (Bilirubin Direk) :
    Bilirubin direk adalah bentuk larut air yang dikeluarkan dari hati. Peningkatan kadar bilirubin direk sering menunjukkan kondisi seperti: Virus hepatitis.
    """)
    st.write("""5. Alkaline_Phosphotase (Fosfatase Alkali) :
    Salah satu enzim hidrolase yang terutama ditemukan pada sebagian besar organ tubuh, terutama dalam jumlah besar di hati, tulang, dan plasenta. Enzim ini berfungsi memindahkan gugus fosfat.
    """)
    st.write("""6. Alamine_Aminotransferase (Transaminase alanina) :
    Alanin transaminase (ALT), atau yang sering dikenal sebagai serum glutamic pyruvic transaminase (SGPT), merupakan enzim yang berguna dalam proses metabolisme protein dalam tubuh. Jika hati tidak berfungsi dengan baik, ALT akan dilepaskan ke dalam darah sehingga kadar ALT dalam darah mengalami peningkatan.
    """)
    st.write("""7. Aspartate_Aminotransferase (Transaminase aspartat) :
    Aspartate aminotransferase merupakan enzim yang terdapat di berbagai jaringan, terutama hati, otot lurik dan otot jantung. Peningkatan aktivitas AST dapat menjadi penanda yang baik adanya kerusakan jaringan lunak.
    """)
    st.write("""8. Total_Protiens (Total Protein) :
    Tes protein total dilakukan untuk mengukur jumlah total dua jenis protein pada tubuh, yaitu albumin dan globulin.
    """)
    st.write("""9. Albumin (Albumin) :
    Albumin adalah cairan infus yang digunakan untuk mengatasi hipoalbuminemia, yaitu rendahnya kadar albumin dalam darah.
    """)
    st.write("""10. Albumin_and_Globulin_Ratio ( Rasio albumin dan globulin) :
    Rasio albumin/globulin adalah perbandingan albumin dengan globulin, yang merupakan konstituen utama protein yang ditemukan dalam darah. Rasio yang abnormal terlihat ketika kadar albumin atau globulin meningkat atau menurun. Rasio abnormal terlihat pada berbagai gangguan, termasuk penyakit ginjal dan hati.
    """)
    st.write("""11. Dataset :
    Dataset ini merupakan output dari prediksi penyakit liver dengan output 1 : Terkena Liver, 2 : Tidak terkena liver.
    """)
    
    st.write("""Mengklasifikasi pasien liver (output) :
    
    1. Output 1    : Menampilkan hasil prediksi Pasien Terkena Liver
    2. Output 2    : Menampilkan hasil prediksi Pasien Tidak Terkena Liver

    """)
    st.write("###### Aplikasi ini untuk : Classification on Indian Liver Patient (Klasifikasi Pasien Liver India) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/Ais-122/Machine-Learning ")

with upload_data:

    df = pd.read_csv('https://raw.githubusercontent.com/Ais-122/Machine-Learning/main/indian_liver_patient.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)

    #Mendefinisikan Varible X dan Y
    X = df[['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens',
            'Albumin','Albumin_and_Globulin_Ratio']]
    y = df['Dataset'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Dataset).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
       
        
    })

    st.write(labels)

    # st.subheader("""Normalisasi Data""")
    # st.write("""Rumus Normalisasi Data :""")
    # st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    # st.markdown("""
    # Dimana :
    # - X = data yang akan dinormalisasi atau data asli
    # - min = nilai minimum semua data asli
    # - max = nilai maksimum semua data asli
    # """)
    # df.weather.value_counts()
    # df = df.drop(columns=["date"])
    # #Mendefinisikan Varible X dan Y
    # X = df.drop(columns=['weather'])
    # y = df['weather'].values
    # df_min = X.min()
    # df_max = X.max()

    
    # #NORMALISASI NILAI X
    # scaler = MinMaxScaler()
    # #scaler.fit(features)
    # #scaler.transform(features)
    # scaled = scaler.fit_transform(X)
    # features_names = X.columns.copy()
    # #features_names.remove('label')
    # scaled_features = pd.DataFrame(scaled, columns=features_names)

    # #Save model normalisasi
    # from sklearn.utils.validation import joblib
    # norm = "normalisasi.save"
    # joblib.dump(scaled_features, norm) 


    # st.subheader('Hasil Normalisasi Data')
    # st.write(scaled_features)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Age = st.number_input('Masukkan umur (Age) : ')
        Gender = st.number_input('Masukkan jenis kelamin berupa angka 1 : Laki-laki, 0 : Perempuan (Gender) : ')
        Total_Bilirubin = st.number_input('Masukkan total bilirubin dalam darah - Berupa angka desimal (Total_Bilirubin) : ')
        Direct_Bilirubin = st.number_input('Masukkan direct bilirubin - Berupa angka desimal (Direct_Bilirubin) : ')
        Alkaline_Phosphotase = st.number_input('Masukkan Alkaline phosphotase - Berupa angka desimal (Alkaline_Phosphotase) : ')
        Alamine_Aminotransferase = st.number_input('Masukkan Alamine Aminotransferase - Berupa angka desimal (Alamine_Aminotransferase) : ')
        Aspartate_Aminotransferase = st.number_input('Masukkan Aspartate Aminotransferase - Berupa angka desimal (Aspartate_Aminotransferase) : ')
        Total_Protiens = st.number_input('Masukkan Total Protiens - Berupa angka desimal (Total_Protiens) : ')
        Albumin = st.number_input('Masukkan ALbumin - Berupa angka desimal (Albumin) : ')
        Albumin_And_Globulin_Ratio = st.number_input('Masukkan Albumin dan Globulin Ratio - Berupa angka desimal (Albumin_And_Globulin_Ratio) : ')
        
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Age,
                Gender,
                Total_Bilirubin,
                Direct_Bilirubin,
                Alkaline_Phosphotase,
                Alamine_Aminotransferase,
                Aspartate_Aminotransferase,
                Total_Protiens,
                Albumin,
                Albumin_And_Globulin_Ratio,
                
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

               
            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
