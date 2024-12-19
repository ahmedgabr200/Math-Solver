function naiveBayesPredict(data, inputFeatures) {
    const classCounts = {};
    const totalCount = data.length;

    // Step 1: Count occurrences of each class
    data.forEach(row => {
        const y = row[3];
        classCounts[y] = (classCounts[y] || 0) + 1;
    });

    const classProbs = {};
    Object.keys(classCounts).forEach(y => {
        classProbs[y] = classCounts[y] / totalCount;
    });

    // Step 2: Compute P(X_i | Y) for each feature
    const featureGivenClass = {};
    Object.keys(classCounts).forEach(y => {
        featureGivenClass[y] = inputFeatures.map((_, j) => {
            const count = data.filter(row => row[3] === y && row[j] === inputFeatures[j]).length;
            return count / classCounts[y];
        });
    });

    // Step 3: Compute posterior probabilities P(Y | X)
    const classLikelihoods = {};
    Object.keys(classCounts).forEach(y => {
        const likelihood = featureGivenClass[y].reduce((product, prob) => product * (prob || 1e-10), 1);
        classLikelihoods[y] = likelihood * classProbs[y];
    });

    // Normalize probabilities
    const totalLikelihood = Object.values(classLikelihoods).reduce((a, b) => a + b, 0);
    Object.keys(classLikelihoods).forEach(y => {
        classLikelihoods[y] /= totalLikelihood;
    });

    // Step 4: Predict class with max likelihood
    const predictedClass = Object.keys(classLikelihoods).reduce((a, b) =>
        classLikelihoods[a] > classLikelihoods[b] ? a : b
    );

    return `Naive Bayes Predicted Class: ${predictedClass}`;
}
