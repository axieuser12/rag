#!/usr/bin/env python3
"""
Security Validator for RAG File Processing System
Scans for potential security issues and API key leakage
"""

import os
import re
import json
from typing import List, Dict, Any

class SecurityValidator:
    def __init__(self):
        self.api_key_patterns = [
            r'sk-[a-zA-Z0-9]{48,}',  # OpenAI API keys
            r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',  # JWT tokens (Supabase)
            r'OPENAI_API_KEY\s*=\s*["\']?sk-[a-zA-Z0-9]{48,}["\']?',  # Hardcoded OpenAI keys
            r'SUPABASE_SERVICE_KEY\s*=\s*["\']?eyJ[a-zA-Z0-9_-]+',  # Hardcoded Supabase keys
        ]
        
        self.sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        
        self.exclude_files = {
            '.env',
            '.env.local',
            '.env.example',
            '__pycache__',
            '.git',
            'node_modules',
            '.vscode',
            '.idea'
        }
    
    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """Scan a single file for security issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
                # Check for API key patterns
                for pattern in self.api_key_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'type': 'api_key_exposure',
                            'line': line_num,
                            'pattern': pattern,
                            'severity': 'HIGH',
                            'description': 'Potential API key found in source code'
                        })
                
                # Check for other sensitive patterns
                for pattern in self.sensitive_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'type': 'sensitive_data',
                            'line': line_num,
                            'pattern': pattern,
                            'severity': 'MEDIUM',
                            'description': 'Potential sensitive data found'
                        })
                        
        except Exception as e:
            issues.append({
                'type': 'scan_error',
                'line': 0,
                'pattern': '',
                'severity': 'LOW',
                'description': f'Error scanning file: {str(e)}'
            })
        
        return {
            'file': file_path,
            'issues': issues
        }
    
    def scan_directory(self, directory: str = '.') -> Dict[str, Any]:
        """Scan entire directory for security issues"""
        all_results = []
        total_issues = 0
        high_severity_count = 0
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_files]
            
            for file in files:
                # Skip excluded files and binary files
                if (file in self.exclude_files or 
                    file.endswith(('.pyc', '.pyo', '.so', '.dll', '.exe', '.jpg', '.png', '.gif', '.pdf'))):
                    continue
                
                file_path = os.path.join(root, file)
                result = self.scan_file(file_path)
                
                if result['issues']:
                    all_results.append(result)
                    total_issues += len(result['issues'])
                    high_severity_count += sum(1 for issue in result['issues'] if issue['severity'] == 'HIGH')
        
        return {
            'scan_summary': {
                'files_scanned': len(all_results),
                'total_issues': total_issues,
                'high_severity_issues': high_severity_count,
                'status': 'FAIL' if high_severity_count > 0 else 'PASS'
            },
            'results': all_results
        }
    
    def generate_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate a human-readable security report"""
        summary = scan_results['scan_summary']
        results = scan_results['results']
        
        report = []
        report.append("=" * 60)
        report.append("SECURITY SCAN REPORT")
        report.append("=" * 60)
        report.append(f"Status: {summary['status']}")
        report.append(f"Files with issues: {summary['files_scanned']}")
        report.append(f"Total issues found: {summary['total_issues']}")
        report.append(f"High severity issues: {summary['high_severity_issues']}")
        report.append("")
        
        if summary['status'] == 'PASS':
            report.append("✅ No high-severity security issues found!")
            report.append("Your code appears to be secure for deployment.")
        else:
            report.append("❌ HIGH SEVERITY ISSUES FOUND!")
            report.append("Please fix these issues before deployment:")
            report.append("")
            
            for result in results:
                high_issues = [issue for issue in result['issues'] if issue['severity'] == 'HIGH']
                if high_issues:
                    report.append(f"File: {result['file']}")
                    for issue in high_issues:
                        report.append(f"  Line {issue['line']}: {issue['description']}")
                    report.append("")
        
        if summary['total_issues'] > summary['high_severity_issues']:
            report.append("Other issues found:")
            for result in results:
                other_issues = [issue for issue in result['issues'] if issue['severity'] != 'HIGH']
                if other_issues:
                    report.append(f"File: {result['file']}")
                    for issue in other_issues:
                        report.append(f"  Line {issue['line']}: {issue['description']} ({issue['severity']})")
                    report.append("")
        
        return "\n".join(report)

def main():
    """Run security validation"""
    validator = SecurityValidator()
    
    print("Running security scan...")
    scan_results = validator.scan_directory()
    
    report = validator.generate_report(scan_results)
    print(report)
    
    # Save report to file
    with open('security_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: security_report.txt")
    
    # Exit with error code if high severity issues found
    if scan_results['scan_summary']['status'] == 'FAIL':
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()