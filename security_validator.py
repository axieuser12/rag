#!/usr/bin/env python3
"""
Security Validator
Scans the project for potential API key leakage and security issues
"""

import os
import re
import glob

class SecurityValidator:
    def __init__(self):
        self.api_key_patterns = [
            r'sk-[a-zA-Z0-9]{48,}',  # OpenAI API keys
            r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',  # JWT tokens (Supabase)
            r'OPENAI_API_KEY\s*=\s*["\']?sk-[a-zA-Z0-9]{48,}',
            r'SUPABASE_SERVICE_KEY\s*=\s*["\']?eyJ[a-zA-Z0-9_-]+',
        ]
        
        self.exclude_files = {
            '.env',
            '.env.example',
            'security_validator.py',
            '__pycache__',
            'node_modules',
            '.git',
            'dist',
            'build'
        }
        
        self.issues = []
    
    def scan_file(self, file_path: str) -> None:
        """Scan a single file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for line_num, line in enumerate(content.splitlines(), 1):
                for pattern in self.api_key_patterns:
                    if re.search(pattern, line):
                        self.issues.append({
                            'file': file_path,
                            'line': line_num,
                            'issue': 'Potential API key exposure',
                            'content': line.strip()[:100] + '...' if len(line.strip()) > 100 else line.strip()
                        })
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    def scan_project(self, root_dir: str = ".") -> None:
        """Scan the entire project for security issues"""
        print("🔍 Scanning project for security issues...")
        
        for root, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_files]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip excluded files and binary files
                if (file in self.exclude_files or 
                    file.endswith(('.pyc', '.jpg', '.png', '.gif', '.pdf', '.zip'))):
                    continue
                
                self.scan_file(file_path)
    
    def generate_report(self) -> None:
        """Generate security report"""
        print("\n" + "="*60)
        print("🛡️  SECURITY SCAN REPORT")
        print("="*60)
        
        if not self.issues:
            print("✅ No security issues found!")
            print("All API keys appear to be properly secured in environment variables.")
        else:
            print(f"⚠️  Found {len(self.issues)} potential security issues:")
            print()
            
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue['issue']}")
                print(f"   File: {issue['file']}")
                print(f"   Line: {issue['line']}")
                print(f"   Content: {issue['content']}")
                print()
            
            print("🔧 RECOMMENDATIONS:")
            print("1. Move all API keys to .env file")
            print("2. Use os.getenv() to read environment variables")
            print("3. Add .env to .gitignore")
            print("4. Never commit API keys to version control")
        
        print("\n" + "="*60)

def main():
    validator = SecurityValidator()
    validator.scan_project()
    validator.generate_report()

if __name__ == "__main__":
    main()