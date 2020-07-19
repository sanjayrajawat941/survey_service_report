import base64
import io

import mysql.connector
from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score


app = Flask(__name__)
app.secret_key = "guest"

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/usr')
def user():
    return render_template('usrsgn.html')

@app.route('/com')
def company():
    return render_template('comsgn.html')
@app.route('/compan')
def comdas():
    return render_template('comdash.html')
@app.route('/user')
def usrdash():
    return render_template('userdash.html')
@app.route('/review')
def surveyform():
    return render_template('review.html')

@app.route('/sgp_code',methods=['post'])
def sgn_up():
    a = str(request.form['t1'])
    b = str(request.form['t2'])
    c = str(request.form['t3'])
    d = str(request.form['t4'])
    e = str(request.form['t5'])

    con = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cur = con.cursor()
    cur.execute("insert into data( name,email,password,address,mobile) values('" + a + "','" + b + "','" + c + "','"+d+"','"+e+"')")
    con.commit()
    return render_template('home.html')
@app.route('/lgp_code',methods=['post'])
def log():


    conn = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cursor = conn.cursor()
    usr=str(request.form['t2'])
    pwd=str(request.form['t3'])

    rb=str(request.form['optradio'])
    if(rb=='user'):

        cursor.execute("select * from data where email='"+usr+"' and password='"+pwd+"'")
        #da = cursor.fetchone()
        if(cursor.fetchone()):

            session["usr"] = usr
            cursor.execute("select * from data where email='" + usr + "'")
            da=cursor.fetchone()
            return render_template('userdash.html',data1=da)
    else:
        cursor.execute("select * from company where email='" + usr + "' and password='" + pwd+ "'")
        if (cursor.fetchone()):
            session["usr"] = usr
            print(usr)
            cursor.execute("select * from company where email='" + usr + "'")
            da = cursor.fetchone()
            print(da)

            return render_template('comdash.html',data1=da)



@app.route('/',methods=['POST'])
def com_up():
    a = str(request.form['s1'])
    b = str(request.form['s2'])
    c = str(request.form['s3'])
    d = str(request.form['s4'])
    e = str(request.form['s5'])
    f = str(request.form['s6'])
    g = str(request.form['s7'])




    con = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cur = con.cursor()
    cur.execute("insert into company( companyname,uname,email,password,address,dept,mobile) values('" + a + "','" + b + "','" + c + "','"+d+"','"+e+"','"+f+"','"+g+"')")
    con.commit()
    return render_template('home.html')

@app.route('/pro',methods=['POST'])
def pro():
    a=str(request.form['t1'])
    b=str(request.form['t2'])
    c=str(session['usr'])
    con = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cur = con.cursor()
    cur.execute( "insert into product(product_name,product_info,company_name) values('" + a + "','" + b + "','" + c + "')")
    con.commit()
    cur.execute("select * from company where email='" + c + "'")
    da = cur.fetchone()
    print(da)

    return render_template('comdash.html', data1=da)
@app.route('/aa')
def display_deals():
    conn = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cursor = conn.cursor()
    c = str(session['usr'])

    try:

        query = "SELECT * from product where company_name='"+c+"'"
        cursor.execute(query)

        data = cursor.fetchall()

        conn.close()



        return render_template("showproduct.html", data=data)

    except Exception as e:
        return (str(e))

@app.route('/delete')

def delete():
        conn = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
        cursor = conn.cursor()
        print(request.args.get('comp_id'))

        try:
            query = "delete from company  where comp_id=" + request.args.get('comp_id')
            cursor.execute(query)
            conn.commit()

            query1 = "SELECT * from company"
            cursor.execute(query1)

            data = cursor.fetchall()

            conn.close()

            # return data

            return render_template("showproduct.html", data=data)
        except Exception as e:
            return (str(e))

@app.route('/bb')
def display_deals2():


    conn = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cursor = conn.cursor()
    query = "SELECT * from product"
    cursor.execute(query)

    data = cursor.fetchall()

    conn.close()

    return render_template("listproduct.html", data1=data)

@app.route('/final', methods=["post"])
def surv():
    conn = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cursor = conn.cursor()
    a=str(request.form['t1'])
    b=str(request.form['t2'])
    c=str(request.form['gender'])
    d=str(request.form['choice'])
    cursor.execute("insert into review(name,age,gender,choice) values('"+a+"','"+b+"','"+c+"','"+d+"')")
    conn.commit()
    return render_template('review.html')

def build_graph(x_coordinates, y_coordinates):
    img=io.BytesIO()
    plt.bar(x_coordinates, y_coordinates)
    plt.savefig(img,format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)
@app.route('/test')
def chart():
    conn = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cursor = conn.cursor()
    df=pd.read_sql('SELECT * FROM review',con=conn)
    print(df.head(5))
    #sns.countplot('age' , 'gender',hue='choice', data=df)
    x=df['age']
    x1=df['gender']
    y=df['choice']
    graph1_url = build_graph(x,y)
    return render_template('result.html',graph1=graph1_url)

'''
@app.route('/report')
def displayR():
    
    df = pd.read_sql('SELECT * FROM review', con=conn)
    x = df.drop(df.columns[[0,3,4]],axis=1,inplace=True)
    y = df[[3]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    logmodel = LogisticRegression()
    logmodel.fit(x_train, y_train)

    predictions = logmodel.predict(x_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    trainingset = df
    # trainingset=file[11:]

    trainingx = df.drop(df.columns[[0,3,4]],axis=1,inplace=True)
    #trainingx = trainingx.astype(float)
    trainingy = df[3]
    # testingx=testingset[:,[1,2,3,4]]
    lr = linear_model.LogisticRegression()
    lr.fit(trainingx, trainingy)
    l = [12,0]
    print(lr.predict([l]))
'''

@app.route('/report')
def display():
    conn = mysql.connector.connect(host="localhost", user="root", password="", db="survey")
    cursor = conn.cursor()
    df=pd.read_sql('SELECT * FROM review', con=conn)
    x=df.drop(['id','name','choice'], axis=1)
    y=df['choice']
    x_t, x_test, y_t


if __name__ == '__main__':
    app.run()
