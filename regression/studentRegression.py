def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    ### your code goes here!
    
    from sklearn import linear_model


    # Create linear regression object
    reg = linear_model.LinearRegression()

    # Train the model using the training sets
    reg.fit(ages_train, net_worths_train)

    # Make predictions using the testing set
    # diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    # print("Coefficients: \n", regr.coef_)
    
    return reg