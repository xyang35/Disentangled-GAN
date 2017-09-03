function receptive_field_size()
% reference: https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m

% compute input size from a given output size
f = @(output_size, ksize, stride) (output_size - 1) * stride + ksize;

out = ...
f(f(f(1, 4, 1), ...   % conv4 -> conv5
             4, 1), ...   % conv3 -> conv4
             4, 2); ...   % conv2 -> conv3
fprintf('n=1 discriminator receptive field size: %d\n', out);

%% n=3 discriminator

% fix the output size to 1 and derive the receptive field in the input
out = ...
f(f(f(f(f(1, 4, 1), ...   % conv4 -> conv5
             4, 1), ...   % conv3 -> conv4
             4, 2), ...   % conv2 -> conv3
             4, 2), ...   % conv1 -> conv2
             4, 2);       % input -> conv1
fprintf('n=3 discriminator receptive field size: %d\n', out);

out = ...
f(f(f(f(f(f(1, 4, 1), ...   % conv4 -> conv5
             4, 1), ...   % conv3 -> conv4
             4, 2), ...   % conv2 -> conv3
             4, 2), ...   % conv1 -> conv2
             4, 2), ...   % conv1 -> conv2
             4, 2);       % input -> conv1
fprintf('n=4 discriminator receptive field size: %d\n', out);

out = ...
f(f(f(f(f(f(f(1, 4, 1), ...   % conv4 -> conv5
             4, 1), ...   % conv3 -> conv4
             4, 2), ...   % conv2 -> conv3
             4, 2), ...   % conv1 -> conv2
             4, 2), ...   % conv1 -> conv2
             4, 2), ...   % conv1 -> conv2
             4, 2);       % input -> conv1
fprintf('n=5 discriminator receptive field size: %d\n', out);
