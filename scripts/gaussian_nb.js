function gaussianNaiveBayes(data, inputFeatures, stdDev) {
    const classStats = {};

    // Step 1: Calculate mean and count for each class
    data.forEach(row => {
        const y = row[3];
        classStats[y] = classStats[y] || { means: [], counts: 0 };
        classStats[y].counts++;
        row.slice(0, -1).forEach((val, j) => {
            classStats[y].means[j] = (classStats[y].means[j] || 0) + val;
        });
    });

    Object.keys(classStats).forEach(y => {
        classStats[y].means = classStats[y].means.map(mean => mean / classStats[y].counts);
    });

    // Step 2: Compute Gaussian likelihoods
    const classLikelihoods = {};
    Object.keys(classStats).forEach(y => {
        const means = classStats[y].means;
        const likelihood = inputFeatures.reduce((product, x, j) => {
            const mean = means[j];
            const variance = Math.pow(stdDev, 2);
            if (variance === 0) return product; // Avoid division by zero
            const gaussian = (1 / Math.sqrt(2 * Math.PI * variance)) * Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
            return product * gaussian;
        }, 1);
        classLikelihoods[y] = likelihood * (classStats[y].counts / data.length);
    });

    // Normalize probabilities
    const totalLikelihood = Object.values(classLikelihoods).reduce((a, b) => a + b, 0);
    Object.keys(classLikelihoods).forEach(y => {
        classLikelihoods[y] /= totalLikelihood;
    });

    // Step 3: Predict class with max likelihood
    const predictedClass = Object.keys(classLikelihoods).reduce((a, b) =>
        classLikelihoods[a] > classLikelihoods[b] ? a : b
    );

    return `Gaussian Naive Bayes Predicted Class: ${predictedClass}`;
}
