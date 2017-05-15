%SIGMOID Compute sigmoid function
%  g = SIGMOID(z) computes the sigmoid of z.

function g = sigmoid(z)
     
    g = zeros(size(z)); % stores sigmoid values of equiv. matrix entry

    for i = 1:size(z,1) % loop through all values in the vector/matrix
        for j = 1:size(z,2)
            g(i,j) = 1 / ( 1 + exp(-z(i,j)) );
        end
    end

end

