# Micrograd

A tiny autograd engine. This is a Ruby implementation of [karpathy/micrograd](https://github.com/karpathy/micrograd). I created this while working through Andrej Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) course.

## Installation

```
gem install micrograd
```

## Usage

Here are some of the operatinos available on `Value`.


```ruby
include Micrograd

a = Value.new(2.0)
b = Value.new(-3.0)
c = Value.new(10.0)
e = a * b
d = e + c
f = Value.new(-2.0)

l = d * f

# Walk through all the values and calculate gradients for them.
l.start_backward
```

Example of training a multi level perceptron (`MLP`):


```ruby
mlp = MLP.new(input_size: 3, layer_sizes: [4, 4, 1])

# These are the training inputs
inputs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

# These are the outputs for each of the inputs above.
desired_outputs = [1.0, -1.0, -1.0, 1.0]

# Training loop
100.times do |n|
  # forward pass
  mlp_outputs = inputs.map { mlp.call(_1).first }
  loss = desired_outputs.zip(mlp_outputs).sum { (_1 - _2) ** 2 }

  # backward pass
  mlp.parameters.each { _1.grad = 0.0 }
  loss.start_backward

  # update the params
  mlp.parameters.each { _1.data -= _1.grad * 0.1 }
end
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/nithinbekal/micrograd. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/nithinbekal/micrograd/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the Micrograd project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/nithinbekal/micrograd/blob/main/CODE_OF_CONDUCT.md).
