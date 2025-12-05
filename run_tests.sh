#!/bin/bash
# Quick test runner for development

echo "Running fast tests (excluding slow/integration tests)..."
python -m pytest tests/ -v -m "not slow" --tb=short

echo ""
echo "Test summary:"
echo "- Fast tests completed"
echo "- To run slow tests: pytest tests/ -v --run-slow"
echo "- To run integration tests: pytest tests/ -v -m integration --run-slow"
