# frozen_string_literal: true

module Micrograd
  class Neuron
    def initialize(n)
      @weights = Array.new(n) { random_value }
      @bias = random_value
    end

    attr_reader :weights, :bias

    def call(xs)
      weights.zip(xs)
        .sum(bias) { |w, x| w * x }
        .tanh
    end

    def parameters = [*weights, bias]

    private

    def random_value
      Value.new(Random.rand(-1.0..1.0)) 
    end
  end
end
