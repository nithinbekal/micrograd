# frozen_string_literal: true

module Micrograd
  class Layer
    def initialize(inputs, outputs)
      @neurons = Array.new(outputs) { Neuron.new(inputs) }
    end

    attr_reader :neurons

    def call(xs)
      @neurons.map { |n| n.call(xs) }
    end

    def parameters = neurons.flat_map(&:parameters)
  end
end
