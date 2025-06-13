def validate_paper_results():
    """Validate that reproduced results match paper claims."""
    
    # Load expected results
    expected_results = load_expected_results("results/paper_results.json")
    
    # Run evaluation
    actual_results = run_full_evaluation()
    
    # Compare results
    validation_report = compare_results(expected_results, actual_results)
    
    # Generate report
    generate_validation_report(validation_report)
    
    return validation_report