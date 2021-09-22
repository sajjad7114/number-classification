from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

print("loading dataset...")
print("It may take some seconds")
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
print("dataset loaded")

x = []
y = []
for i in range(len(Y)):
    if Y[i] == '2' or Y[i] == '3' or Y[i] == '7':
        y.append(Y[i])
        x.append(X[i])


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.1, random_state=42)
X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, train_size=0.5, random_state=42)
c = [1, 10, 100, 1000, 10000]
min_error_lovo = 0
min_c_lovo = 0
print("proseccing one vs. one linear SVM .....", end="")
cc = 1
for i in c:
    svclassifier = SVC(kernel='linear', C=i, decision_function_shape='ovo')
    svclassifier.fit(X_train2, y_train2)
    y_pred = svclassifier.predict(X_valid)
    coef_matrix = confusion_matrix(y_valid, y_pred)
    error = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
    if min_c_lovo == 0:
        min_c_lovo = i
        min_error_lovo = error
    else:
        if min_error_lovo > error:
            min_error_lovo = error
            min_c_lovo = i
    print("\b\b\b\b\b", end="")
    for j in range(cc):
        print("*", end="")
    for j in range(5 - cc):
        print(".", end="")
    cc += 1

svclassifier = SVC(kernel='linear', C=min_c_lovo, decision_function_shape='ovo')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
coef_matrix = confusion_matrix(y_test, y_pred)
error_lovo = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
print("\nfor one vs. one linear SVM we have optimal C = "+str(min_c_lovo)+" and test error at this C is : "+str(error_lovo/len(y_test)))

min_error_lovr = 0
min_c_lovr = 0
print("proseccing one vs. all linear SVM .....", end="")
cc = 1
for i in c:
    svclassifier = SVC(kernel='linear', C=i, decision_function_shape='ovr')
    svclassifier.fit(X_train2, y_train2)
    y_pred = svclassifier.predict(X_valid)
    coef_matrix = confusion_matrix(y_valid, y_pred)
    error = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
    if min_c_lovr == 0:
        min_c_lovr = i
        min_error_lovr = error
    else:
        if min_error_lovr > error:
            min_error_lovr = error
            min_c_lovr = i
    print("\b\b\b\b\b", end="")
    for j in range(cc):
        print("*", end="")
    for j in range(5 - cc):
        print(".", end="")
    cc += 1

svclassifier = SVC(kernel='linear', C=min_c_lovr, decision_function_shape='ovr')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
coef_matrix = confusion_matrix(y_test, y_pred)
error_lovr = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
print("\nfor one vs. all linear SVM we have optimal C = "+str(min_c_lovr)+" and test error at this C is : "+str(error_lovr/len(y_test)))

if error_lovo == error_lovr:
    if min_c_lovr == min_c_lovo:
        print("one vs. one and one vs. all linear SVM have same error on same optimal C")
    else:
        print("one vs. one and one vs. all linear SVM have same error on different optimal C")
elif error_lovo > error_lovr:
    print("one vs. all linear SVM predicts better than one vs. one linear SVM")
else:
    print("one vs. one linear SVM predicts better than one vs. all linear SVM")

gamma = [0.01, 0.1, 1, 10, 100]

min_error_govo = 0
min_c_govo = 0
min_gamma_govo = 0
print("proseccing one vs. one gaussian SVM .........................", end="")
cc = 1
for i in c:
    gg = 1
    for g in gamma:
        svclassifier = SVC(kernel='rbf', C=i, gamma=g, decision_function_shape='ovo', random_state=0)
        svclassifier.fit(X_train2, y_train2)
        y_pred = svclassifier.predict(X_valid)
        coef_matrix = confusion_matrix(y_valid, y_pred)
        error = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
        if min_c_govo == 0:
            min_c_govo = i
            min_error_govo = error
            min_gamma_govo = g
        else:
            if min_error_govo > error:
                min_error_govo = error
                min_c_govo = i
                min_gamma_govo = g
        print("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b", end="")
        for j in range(5*(cc-1)+gg):
            print("*", end="")
        for j in range(25 - (5*(cc-1)+gg)):
            print(".", end="")
        gg += 1
    cc += 1

svclassifier = SVC(kernel='rbf', C=min_c_govo, gamma=min_gamma_govo, decision_function_shape='ovo', random_state=0)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
coef_matrix = confusion_matrix(y_test, y_pred)
error_govo = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
print("\nfor one vs. one gaussian SVM we have optimal C = "+str(min_c_govo), end="")
print(" and optimal gamma = " + str(min_gamma_govo), end="")
print(" test error at this C and gamma is : "+str(error_govo/len(y_test)))

min_error_govr = 0
min_c_govr = 0
min_gamma_govr = 0
print("proseccing one vs. all gaussian SVM .........................", end="")
cc = 1
for i in c:
    gg = 1
    for g in gamma:
        svclassifier = SVC(kernel='rbf', C=i, gamma=g, decision_function_shape='ovr', random_state=0)
        svclassifier.fit(X_train2, y_train2)
        y_pred = svclassifier.predict(X_valid)
        coef_matrix = confusion_matrix(y_valid, y_pred)
        error = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
        if min_c_govr == 0:
            min_c_govr = i
            min_error_govr = error
            min_gamma_govr = g
        else:
            if min_error_govr > error:
                min_error_govr = error
                min_c_govr = i
                min_gamma_govr = g
        print("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b", end="")
        for j in range(5*(cc-1)+gg):
            print("*", end="")
        for j in range(25 - (5*(cc-1)+gg)):
            print(".", end="")
        gg += 1
    cc += 1

svclassifier = SVC(kernel='rbf', C=min_c_govr, gamma=min_gamma_govr, decision_function_shape='ovr', random_state=0)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
coef_matrix = confusion_matrix(y_test, y_pred)
error_govr = coef_matrix[0][1]+coef_matrix[1][0]+coef_matrix[0][2]+coef_matrix[2][0]+coef_matrix[2][1]+coef_matrix[1][2]
print("\nfor one vs. all gaussian SVM we have optimal C = "+str(min_c_govr), end="")
print(" and optimal gamma = " + str(min_gamma_govr), end="")
print(" test error at this C and gamma is : "+str(error_govr/len(y_test)))

if error_govo == error_govr:
    if min_c_govr == min_c_govo:
        if min_gamma_govr == min_gamma_govo:
            print("one vs. one and one vs. all gaussian SVM have same error on same optimal C and same optimal gamma")
        else:
            print("one vs. one and one vs. all gaussian SVM have same error on same optimal C but different optimal gamma")
    else:
        if min_gamma_govr == min_gamma_govo:
            print("one vs. one and one vs. all gaussian SVM have same error on same optimal gamma but different optimal C")
        else:
            print("one vs. one and one vs. all gaussian SVM have same error on different optimal C and different optimal gamma")
elif error_govo > error_govr:
    print("one vs. all gaussian SVM predicts better than one vs. one gaussian SVM")
else:
    print("one vs. one gaussian SVM predicts better than one vs. all gaussian SVM")
