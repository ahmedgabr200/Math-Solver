function linearRegressionPredict(data, inputX) {
    const X = data.map(row => row[0]);
    const Y = data.map(row => row[3]);

    const n = X.length;
    const meanX = X.reduce((sum, val) => sum + val, 0) / n;
    const meanY = Y.reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
        numerator += (X[i] - meanX) * (Y[i] - meanY);
        denominator += (X[i] - meanX) ** 2;
    }

    const slope = numerator / denominator;
    const intercept = meanY - slope * meanX;

    const prediction = slope * inputX + intercept;

    return `Linear Regression Equation: y = ${slope.toFixed(2)}x + ${intercept.toFixed(2)}
Predicted Value for x = ${inputX}: ${prediction.toFixed(2)}`;
}
