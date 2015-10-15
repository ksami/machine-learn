
// input dataset
var input = [
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
];

// output dataset
var output = [
    0,
    0,
    0,
    1,
];

// sigmoid function
function nonlin(x, deriv){
    if(deriv === true){
        return x*(1-x);
    }
    return 1/(1+Math.exp(-x));
}


// seed random numbers to make calc deterministic
// obtain same results each time to compare
var seed = 1;
function random(){
    var n = Math.sin(seed++) * 10000;
    return n - Math.floor(n);
}

var l0 = [];
var l1 = [];
var l1_error = [];
var l1_delta = [];

// initialize weights randomly with mean 0
var syn0 = [];
syn0[0] = 2*random()-1;
syn0[1] = 2*random()-1;
syn0[2] = 2*random()-1;


// n iterations
for (var i = 0; i < 1000000; i++) {

    // for each set of data
    for (var j = 0; j < input.length; j++) {
        // forward propagation
        l0 = input;

        // multiply each data set in l0 by syn0
        // representing network's prediction
        // aim is to have l1[j] == output[j]
        // l0[j] = [x,y,z]; syn0 = [a,b,c]
        // l1[j] = a*x + b*y + c*z
        l1[j] = (l0[j][0] * syn0[0]) + (l0[j][1] * syn0[1]) + (l0[j][2] * syn0[2]);

        // maps result to value in the range 0-1
        l1[j] = nonlin(l1[j]);
     
        // how far off is the prediction?
        l1_error[j] = output[j] - l1[j];
        
        // multiply how much we missed by the
        // slope of the sigmoid at the values in l1
        // force predictions away from centre
        // be more decisive!
        l1_delta[j] = l1_error[j] * nonlin(l1[j], true);

        // update weights a,b,c
        syn0[0] += l0[j][0] * l1_delta[j];
        syn0[1] += l0[j][1] * l1_delta[j];
        syn0[2] += l0[j][2] * l1_delta[j];
    }

    if(i%10000 === 0) {
        console.log(l1);
    }
}

console.log(l1);
console.log("\nAfter training, test with [1,1,1]:");
result = nonlin(syn0[0] + syn0[1] + syn0[2]);
console.log(result);