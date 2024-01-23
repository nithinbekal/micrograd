# frozen_string_literal: true

module Micrograd
  class Neuron
    def initialize(n)
      @weights = Array.new(n) { random_value }
      @bias = random_value
    end

    def call(xs)
      @weights.zip(xs)
        .sum(@bias) { |w, x| w * x }
        .tanh
    end

    private

    def random_value
      Value.new(Random.rand(-1.0..1.0)) 
    end
  end
end
