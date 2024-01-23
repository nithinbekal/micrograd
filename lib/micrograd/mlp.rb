# frozen_string_literal: true

module Micrograd
  # Multi layer perceptron
  #
  #     MLP.new(input_size: 3, layer_sizes: [4, 4, 1])
  #
  # In the above example, we have 3 inputs, an we have 3 layers, two layers of
  # 4, and one output layer.
  #
  class MLP
    def initialize(input_size:, layer_sizes:)
      @layers = [input_size, *layer_sizes]
        .each_cons(2)
        .map { |x, y| Layer.new(x, y) }
    end

    attr_reader :layers

    def call(xs)
      @layers.each { |layer| xs = layer.call(xs) }
      xs
    end

    def parameters = layers.flat_map(&:parameters)
  end
end
