"""Лабораторная работа №1: MLP (NumPy) и логистическая регрессия.
Запуск: pip install numpy pandas scikit-learn matplotlib; python lab1_mlp.py
Датасет UCI загружается при запуске, требуется интернет.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay

SEED=42
URL='https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
COLUMNS=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
NUM=['age','trestbps','chol','thalach','oldpeak']
CAT=['sex','cp','fbs','restecg','exang','slope','ca','thal']
OUT=Path('lab1_results'); OUT.mkdir(exist_ok=True)

class MLP:
    def __init__(self,input_size,hidden_size,output_size=2,learning_rate=.05,seed=42):
        rng=np.random.default_rng(seed)
        self.lr=learning_rate
        self.W1=rng.standard_normal((input_size,hidden_size))*np.sqrt(2/input_size)
        self.b1=np.zeros((1,hidden_size))
        self.W2=rng.standard_normal((hidden_size,output_size))*np.sqrt(2/hidden_size)
        self.b2=np.zeros((1,output_size))
    def forward(self,X):
        self.z1=X@self.W1+self.b1
        self.a1=np.maximum(0,self.z1)
        self.z2=self.a1@self.W2+self.b2
        z=self.z2-self.z2.max(axis=1,keepdims=True)
        e=np.exp(z)
        self.a2=e/e.sum(axis=1,keepdims=True)
        return self.a2
    def compute_loss(self,y_pred,y_true):
        return -np.mean(np.sum(y_true*np.log(np.clip(y_pred,1e-9,1.)),axis=1))
    def backward(self,X,y_true):
        m=len(X)
        dz2=self.a2-y_true
        dW2=self.a1.T@dz2/m
        db2=dz2.sum(axis=0,keepdims=True)/m
        dz1=(dz2@self.W2.T)*(self.z1>0)
        dW1=X.T@dz1/m
        db1=dz1.sum(axis=0,keepdims=True)/m
        self.W1-=self.lr*dW1
        self.b1-=self.lr*db1
        self.W2-=self.lr*dW2
        self.b2-=self.lr*db2
    def predict_proba(self,X): return self.forward(X)[:,1]
    def predict(self,X): return (self.predict_proba(X)>=.5).astype(int)

def train(model,X,yoh,Xv,yv,epochs=500,batch_size=32):
    hist={'train_loss':[],'val_auc':[]}
    rng=np.random.default_rng(SEED)
    for epoch in range(epochs):
        idx=rng.permutation(len(X))
        losses=[]
        for start in range(0,len(X),batch_size):
            b=idx[start:start+batch_size]
            yp=model.forward(X[b]);losses.append(model.compute_loss(yp,yoh[b]));model.backward(X[b],yoh[b])
        hist['train_loss'].append(float(np.mean(losses)))
        hist['val_auc'].append(float(roc_auc_score(yv,model.predict_proba(Xv))))
    return hist

def metrics(y,p,prob):
    return dict(Accuracy=accuracy_score(y,p),Precision=precision_score(y,p,zero_division=0),Recall=recall_score(y,p,zero_division=0),F1=f1_score(y,p,zero_division=0),ROC_AUC=roc_auc_score(y,prob))

def main():
    raw=pd.read_csv(URL,header=None,names=COLUMNS,na_values='?')
    print('Исходная размерность:',raw.shape,'\nПропуски:\n',raw.isna().sum().to_string())
    df=raw.dropna().copy();df['target']=(df['target']>0).astype(int)
    print('После очистки:',df.shape,'\nКлассы:\n',df['target'].value_counts().sort_index().to_string())
    Xnum=df[NUM].to_numpy(dtype=float)
    Xcat=pd.get_dummies(df[CAT].astype('category'),drop_first=True,dtype=float).to_numpy()
    X=np.hstack([Xnum,Xcat]);y=df.target.to_numpy()
    original_indices=df.index.to_numpy()
    Xtr,Xtemp,ytr,ytemp,itr,itemp=train_test_split(X,y,original_indices,test_size=.30,random_state=42,stratify=y)
    Xv,Xte,yv,yte,iv,ite=train_test_split(Xtemp,ytemp,itemp,test_size=.5,random_state=42,stratify=ytemp)
    scaler=StandardScaler();n=len(NUM)
    Xtr[:,:n]=scaler.fit_transform(Xtr[:,:n]);Xv[:,:n]=scaler.transform(Xv[:,:n]);Xte[:,:n]=scaler.transform(Xte[:,:n])
    Yoh=np.eye(2)[ytr]
    print(f'Размерность X={X.shape}; train/val/test={len(Xtr)}/{len(Xv)}/{len(Xte)}')
    trials=[]; histories={};models={}
    for hidden,lr in [(8,.1),(16,.1),(32,.05)]:
        model=MLP(X.shape[1],hidden,learning_rate=lr,seed=SEED)
        hist=train(model,Xtr,Yoh,Xv,yv)
        val=roc_auc_score(yv,model.predict_proba(Xv))
        # Тестовые значения приведены для полноты таблицы, но НЕ используются при выборе модели.
        test_auc=roc_auc_score(yte,model.predict_proba(Xte))
        trials.append({'hidden_size':hidden,'learning_rate':lr,'val_auc':val,'test_auc':test_auc})
        histories[(hidden,lr)]=hist;models[(hidden,lr)]=model
        print('Эксперимент:',hidden,lr,'Validation AUC=',round(val,4))
    best=max(trials,key=lambda t:t['val_auc']);key=(best['hidden_size'],best['learning_rate'])
    model=models[key];p=model.predict_proba(Xte);pred=(p>=.5).astype(int)
    baseline=LogisticRegression(max_iter=1000,random_state=42).fit(Xtr,ytr)
    pb=baseline.predict_proba(Xte)[:,1];predb=baseline.predict(Xte)
    pd.DataFrame(trials).to_csv(OUT/'hyperparameters.csv',index=False)
    scores=pd.DataFrame({'MLP':metrics(yte,pred,p),'LogisticRegression':metrics(yte,predb,pb)})
    scores.to_csv(OUT/'metrics.csv');print('\nВыбрано:',best,'\n',scores.to_string());print('\nМатрица ошибок MLP:\n',confusion_matrix(yte,pred))
    hist=histories[key]
    fig,ax=plt.subplots(1,2,figsize=(11,4));ax[0].plot(hist['train_loss']);ax[0].set(xlabel='Epoch',ylabel='Loss',title='Train loss');ax[1].plot(hist['val_auc']);ax[1].set(xlabel='Epoch',ylabel='ROC-AUC',title='Validation ROC-AUC');fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));RocCurveDisplay.from_predictions(yte,p,name='MLP',ax=ax);RocCurveDisplay.from_predictions(yte,pb,name='Logistic regression',ax=ax);ax.plot([0,1],[0,1],'--',label='Random');ax.legend();fig.tight_layout();fig.savefig(OUT/'roc.png',dpi=180);plt.close(fig)
    cm=confusion_matrix(yte,pred);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Actual',title='MLP confusion matrix')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16)
    fig.tight_layout();fig.savefig(OUT/'confusion.png',dpi=180);plt.close(fig)
    mistakes=(pred!=yte)&(predb==yte); reverse=(pred==yte)&(predb!=yte)
    detail=df.loc[ite].copy();detail['MLP_prediction']=pred;detail['MLP_probability']=p;detail['LogReg_prediction']=predb;detail['LogReg_probability']=pb
    detail['MLP_wrong_LogReg_correct']=mistakes;detail['MLP_correct_LogReg_wrong']=reverse
    detail.to_csv(OUT/'patient_predictions.csv',index_label='Original_index')
    print('MLP ошиблась, LR верна:',int(mistakes.sum()),'обратных:',int(reverse.sum()))
    if len(np.where(mistakes)[0]): print(detail.iloc[np.where(mistakes)[0][:5]][['target','MLP_prediction','MLP_probability','LogReg_prediction']].to_string())
    print('Готово. Результаты сохранены:',OUT.resolve())
if __name__=='__main__':main()
