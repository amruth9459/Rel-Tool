#!/usr/bin/env python3
"""
WhatsApp Multi-Export Merger & Chronological Organizer
Handles multiple chat exports, deduplicates, and orders chronologically
Perfect for 200,000+ messages across multiple exports
"""

import os
import re
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import json
from dataclasses import dataclass
from tqdm import tqdm
import argparse
import pickle

@dataclass
class ParsedMessage:
    """Structured message with all metadata"""
    timestamp: datetime
    sender: str
    text: str
    raw_line: str
    export_source: str
    message_hash: str
    
    def __hash__(self):
        return hash(self.message_hash)
    
    def __eq__(self, other):
        return self.message_hash == other.message_hash

class WhatsAppMultiExportMerger:
    """
    Merge multiple WhatsApp exports intelligently
    Handles duplicates, different formats, and maintains chronological order
    """
    
    def __init__(self):
        # Multiple date format patterns WhatsApp uses
        self.date_patterns = [
            # US format: MM/DD/YY, HH:MM AM/PM
            (r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.*)',
             '%m/%d/%y, %I:%M %p'),
            # 24-hour format: DD/MM/YYYY, HH:MM
            (r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)',
             '%d/%m/%Y, %H:%M'),
            # Brackets format: [MM/DD/YY, HH:MM:SS AM/PM]
            (r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.*)',
             '%m/%d/%y, %I:%M:%S %p'),
            # European format: DD.MM.YYYY, HH:MM
            (r'(\d{1,2}\.\d{1,2}\.\d{4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)',
             '%d.%m.%Y, %H:%M'),
            # ISO-like format: YYYY-MM-DD HH:MM:SS
            (r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s-\s([^:]+):\s(.*)',
             '%Y-%m-%d %H:%M:%S'),
        ]
        
        self.messages = []
        self.duplicates_removed = 0
        self.total_processed = 0
        self.export_stats = {}
        
    def parse_single_export(self, file_path: str, export_name: str = None) -> List[ParsedMessage]:
        """Parse a single WhatsApp export file"""
        if export_name is None:
            export_name = Path(file_path).stem
            
        print(f"📖 Parsing {export_name}...")
        
        messages = []
        current_message = None
        lines_processed = 0
        
        # Detect encoding
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.readlines()
                print(f"  ✅ Detected encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
                
        if content is None:
            print(f"  ❌ Could not read file with any encoding")
            return []
        
        # Parse messages
        for line_num, line in enumerate(tqdm(content, desc=f"Processing {export_name}")):
            lines_processed += 1
            matched = False
            
            # Try each date pattern
            for pattern, date_format in self.date_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous message if exists
                    if current_message:
                        messages.append(current_message)
                    
                    # Parse new message
                    timestamp_str = match.group(1)
                    sender = match.group(2).strip()
                    text = match.group(3).strip()
                    
                    # Parse timestamp
                    try:
                        timestamp = datetime.strptime(timestamp_str, date_format)
                    except ValueError:
                        # Try alternative formats
                        for alt_format in ['%m/%d/%Y, %I:%M %p', '%d/%m/%y, %H:%M']:
                            try:
                                timestamp = datetime.strptime(timestamp_str, alt_format)
                                break
                            except:
                                continue
                        else:
                            # Use current time as fallback
                            timestamp = datetime.now()
                    
                    # Create message hash for deduplication
                    message_content = f"{timestamp.isoformat()}|{sender}|{text}"
                    message_hash = hashlib.md5(message_content.encode()).hexdigest()
                    
                    current_message = ParsedMessage(
                        timestamp=timestamp,
                        sender=sender,
                        text=text,
                        raw_line=line.strip(),
                        export_source=export_name,
                        message_hash=message_hash
                    )
                    matched = True
                    break
            
            # If not matched, might be continuation of previous message
            if not matched and current_message:
                current_message.text += '\n' + line.strip()
                # Update hash
                message_content = f"{current_message.timestamp.isoformat()}|{current_message.sender}|{current_message.text}"
                current_message.message_hash = hashlib.md5(message_content.encode()).hexdigest()
        
        # Add last message
        if current_message:
            messages.append(current_message)
        
        print(f"  ✅ Parsed {len(messages)} messages from {lines_processed} lines")
        
        self.export_stats[export_name] = {
            'messages': len(messages),
            'lines': lines_processed,
            'date_range': (
                min(m.timestamp for m in messages) if messages else None,
                max(m.timestamp for m in messages) if messages else None
            )
        }
        
        return messages
    
    def merge_exports(self, export_files: List[str]) -> pd.DataFrame:
        """Merge multiple exports into single chronological dataset"""
        print(f"\n🔄 Merging {len(export_files)} exports...")
        
        all_messages = []
        
        # Parse each export
        for i, export_file in enumerate(export_files, 1):
            if not os.path.exists(export_file):
                print(f"  ⚠️ File not found: {export_file}")
                continue
                
            export_name = f"Export_{i}_{Path(export_file).stem}"
            messages = self.parse_single_export(export_file, export_name)
            all_messages.extend(messages)
            self.total_processed += len(messages)
        
        print(f"\n📊 Total messages collected: {len(all_messages)}")
        
        # Remove duplicates
        unique_messages = self.remove_duplicates(all_messages)
        
        # Sort chronologically
        unique_messages.sort(key=lambda x: x.timestamp)
        
        # Convert to DataFrame
        df = self.messages_to_dataframe(unique_messages)
        
        # Add analysis metadata
        df = self.add_metadata(df)
        
        print(f"\n✅ Merged dataset ready:")
        print(f"  • Total messages: {len(df)}")
        print(f"  • Duplicates removed: {self.duplicates_removed}")
        print(f"  • Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  • Participants: {', '.join(df['sender'].unique())}")
        
        return df
    
    def remove_duplicates(self, messages: List[ParsedMessage]) -> List[ParsedMessage]:
        """Remove duplicate messages based on content hash"""
        print("\n🔍 Removing duplicates...")
        
        seen_hashes = set()
        unique_messages = []
        
        for msg in tqdm(messages, desc="Deduplicating"):
            if msg.message_hash not in seen_hashes:
                seen_hashes.add(msg.message_hash)
                unique_messages.append(msg)
            else:
                self.duplicates_removed += 1
        
        print(f"  ✅ Removed {self.duplicates_removed} duplicates")
        
        # Also remove near-duplicates (same sender, similar time, similar text)
        final_messages = []
        last_msg = None
        
        for msg in unique_messages:
            if last_msg:
                time_diff = abs((msg.timestamp - last_msg.timestamp).total_seconds())
                text_similarity = self.calculate_similarity(msg.text, last_msg.text)
                
                # Skip if very similar message from same sender within 5 seconds
                if (msg.sender == last_msg.sender and 
                    time_diff < 5 and 
                    text_similarity > 0.9):
                    self.duplicates_removed += 1
                    continue
                    
            final_messages.append(msg)
            last_msg = msg
        
        return final_messages
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (0-1)"""
        if text1 == text2:
            return 1.0
        
        # Simple character overlap
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def messages_to_dataframe(self, messages: List[ParsedMessage]) -> pd.DataFrame:
        """Convert messages to pandas DataFrame"""
        data = []
        
        for msg in messages:
            data.append({
                'timestamp': msg.timestamp,
                'sender': msg.sender,
                'message': msg.text,
                'export_source': msg.export_source,
                'message_hash': msg.message_hash,
                'date': msg.timestamp.date(),
                'time': msg.timestamp.time(),
                'hour': msg.timestamp.hour,
                'day_of_week': msg.timestamp.strftime('%A'),
                'month': msg.timestamp.strftime('%B'),
                'year': msg.timestamp.year
            })
        
        return pd.DataFrame(data)
    
    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add analytical metadata to merged dataset"""
        print("\n📈 Adding analytical metadata...")
        
        # Message length
        df['message_length'] = df['message'].str.len()
        
        # Response time (time since last message from different sender)
        df['prev_sender'] = df['sender'].shift(1)
        df['prev_timestamp'] = df['timestamp'].shift(1)
        df['is_response'] = df['sender'] != df['prev_sender']
        df['response_time_minutes'] = (
            (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 60
        ).where(df['is_response'])
        
        # Conversation sessions (new session after 6 hours gap)
        df['time_gap_hours'] = (
            (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600
        )
        df['new_session'] = df['time_gap_hours'] > 6
        df['session_id'] = df['new_session'].cumsum()
        
        # Message order in conversation
        df['message_index'] = range(len(df))
        df['message_position_in_session'] = df.groupby('session_id').cumcount()
        
        # Context window (10 messages before and after)
        window_size = 10
        df['context_start'] = (df['message_index'] - window_size).clip(lower=0)
        df['context_end'] = (df['message_index'] + window_size).clip(upper=len(df)-1)
        
        # Clean up temporary columns
        df = df.drop(columns=['prev_sender', 'prev_timestamp', 'time_gap_hours'])
        
        return df
    
    def detect_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect potential conflicts in merged exports"""
        print("\n⚠️ Checking for timeline conflicts...")
        
        conflicts = []
        
        # Check for messages with identical timestamps from different sources
        duplicate_times = df[df.duplicated(subset=['timestamp', 'sender'], keep=False)]
        if not duplicate_times.empty:
            conflicts.append({
                'type': 'duplicate_timestamps',
                'count': len(duplicate_times),
                'affected_sources': duplicate_times['export_source'].unique().tolist()
            })
        
        # Check for timeline gaps
        df_sorted = df.sort_values('timestamp')
        gaps = []
        
        for i in range(1, len(df_sorted)):
            time_diff = (df_sorted.iloc[i]['timestamp'] - df_sorted.iloc[i-1]['timestamp']).total_seconds() / 3600
            if time_diff > 24 * 7:  # Gap of more than a week
                gaps.append({
                    'start': df_sorted.iloc[i-1]['timestamp'],
                    'end': df_sorted.iloc[i]['timestamp'],
                    'gap_days': time_diff / 24
                })
        
        if gaps:
            conflicts.append({
                'type': 'timeline_gaps',
                'count': len(gaps),
                'gaps': gaps[:5]  # Show first 5 gaps
            })
        
        if conflicts:
            print("  ⚠️ Conflicts detected:")
            for conflict in conflicts:
                print(f"    • {conflict['type']}: {conflict['count']} instances")
        else:
            print("  ✅ No timeline conflicts detected")
        
        return df
    
    def save_merged_export(self, df: pd.DataFrame, output_path: str = None):
        """Save merged and organized export"""
        if output_path is None:
            output_path = f"merged_whatsapp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save as multiple formats
        base_path = Path(output_path).stem
        
        # 1. Pickle (fastest for Python)
        pickle_path = f"{base_path}.pkl"
        df.to_pickle(pickle_path)
        print(f"\n💾 Saved to {pickle_path} (for fast loading)")
        
        # 2. CSV (for compatibility)
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved to {csv_path} (for Excel/spreadsheets)")
        
        # 3. Recreate WhatsApp format
        txt_path = f"{base_path}_reconstructed.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                timestamp_str = row['timestamp'].strftime('%m/%d/%Y, %I:%M %p')
                f.write(f"{timestamp_str} - {row['sender']}: {row['message']}\n")
        print(f"💾 Saved to {txt_path} (WhatsApp format)")
        
        # 4. Statistics report
        stats_path = f"{base_path}_statistics.json"
        stats = {
            'total_messages': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'participants': df['sender'].unique().tolist(),
            'messages_per_participant': df['sender'].value_counts().to_dict(),
            'exports_merged': list(self.export_stats.keys()),
            'export_details': self.export_stats,
            'duplicates_removed': self.duplicates_removed,
            'sessions_detected': df['session_id'].max(),
            'average_message_length': df['message_length'].mean(),
            'average_response_time_minutes': df['response_time_minutes'].mean()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"💾 Saved statistics to {stats_path}")
        
        return base_path


class ChronologicalAnalyzer:
    """
    Analyze merged chronological data with context preservation
    """
    
    def __init__(self, merged_df: pd.DataFrame):
        self.df = merged_df
        self.total_messages = len(merged_df)
        
    def analyze_relationship_progression(self) -> Dict:
        """Analyze how relationship evolved chronologically"""
        print("\n📊 Analyzing relationship progression...")
        
        # Divide into time periods
        total_days = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        
        if total_days > 365:
            # Analyze by month
            self.df['period'] = self.df['timestamp'].dt.to_period('M')
        elif total_days > 30:
            # Analyze by week
            self.df['period'] = self.df['timestamp'].dt.to_period('W')
        else:
            # Analyze by day
            self.df['period'] = self.df['timestamp'].dt.to_period('D')
        
        progression = {}
        
        for period in self.df['period'].unique():
            period_data = self.df[self.df['period'] == period]
            
            progression[str(period)] = {
                'messages': len(period_data),
                'participants': period_data['sender'].value_counts().to_dict(),
                'avg_message_length': period_data['message_length'].mean(),
                'avg_response_time': period_data['response_time_minutes'].mean(),
                'sessions': period_data['session_id'].nunique()
            }
        
        return progression
    
    def find_critical_periods(self) -> List[Dict]:
        """Identify critical periods in the relationship"""
        print("\n🔍 Identifying critical periods...")
        
        critical_periods = []
        
        # 1. High activity periods
        daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
        threshold = daily_counts.quantile(0.9)
        high_activity_days = daily_counts[daily_counts > threshold]
        
        for date, count in high_activity_days.items():
            critical_periods.append({
                'date': date,
                'type': 'high_activity',
                'messages': count,
                'description': f"Unusually high activity: {count} messages"
            })
        
        # 2. Long silence periods
        self.df['next_timestamp'] = self.df['timestamp'].shift(-1)
        self.df['gap_hours'] = (
            (self.df['next_timestamp'] - self.df['timestamp']).dt.total_seconds() / 3600
        )
        
        long_gaps = self.df[self.df['gap_hours'] > 48]  # Gaps > 2 days
        
        for _, row in long_gaps.iterrows():
            critical_periods.append({
                'date': row['timestamp'],
                'type': 'silence',
                'duration_hours': row['gap_hours'],
                'description': f"Silence for {row['gap_hours']:.1f} hours"
            })
        
        # Sort by date
        critical_periods.sort(key=lambda x: x['date'])
        
        return critical_periods[:20]  # Return top 20 critical periods
    
    def analyze_communication_balance(self) -> Dict:
        """Analyze communication balance over time"""
        print("\n⚖️ Analyzing communication balance...")
        
        # Calculate rolling balance
        window = min(100, len(self.df) // 100)  # Adaptive window size
        
        balance_analysis = {
            'overall': {},
            'by_period': {},
            'imbalance_periods': []
        }
        
        # Overall statistics
        sender_counts = self.df['sender'].value_counts()
        total = sender_counts.sum()
        
        for sender, count in sender_counts.items():
            balance_analysis['overall'][sender] = {
                'messages': count,
                'percentage': (count / total) * 100,
                'avg_message_length': self.df[self.df['sender'] == sender]['message_length'].mean()
            }
        
        # Balance ratio (ideal is 1.0 for 50/50 split)
        if len(sender_counts) == 2:
            balance_ratio = sender_counts.max() / sender_counts.min()
            balance_analysis['balance_ratio'] = balance_ratio
            balance_analysis['balanced'] = balance_ratio < 1.5  # Less than 60/40 split
        
        return balance_analysis
    
    def extract_conversation_threads(self) -> List[pd.DataFrame]:
        """Extract coherent conversation threads"""
        print("\n🧵 Extracting conversation threads...")
        
        threads = []
        
        for session_id in self.df['session_id'].unique()[:100]:  # Limit for performance
            session = self.df[self.df['session_id'] == session_id]
            
            if len(session) >= 5:  # Only meaningful conversations
                threads.append({
                    'session_id': session_id,
                    'start': session['timestamp'].min(),
                    'end': session['timestamp'].max(),
                    'duration_minutes': (
                        (session['timestamp'].max() - session['timestamp'].min()).total_seconds() / 60
                    ),
                    'messages': len(session),
                    'participants': session['sender'].unique().tolist(),
                    'data': session
                })
        
        # Sort by message count (most significant conversations)
        threads.sort(key=lambda x: x['messages'], reverse=True)
        
        return threads


def create_merged_analyzer_pipeline(export_files: List[str], output_dir: str = "analysis_output"):
    """
    Complete pipeline: Merge → Organize → Analyze
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     WhatsApp Multi-Export Chronological Analyzer              ║
    ║           Merging, Deduplicating, and Analyzing               ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Merge exports
    merger = WhatsAppMultiExportMerger()
    merged_df = merger.merge_exports(export_files)
    
    # Step 2: Check for conflicts
    merged_df = merger.detect_conflicts(merged_df)
    
    # Step 3: Save merged data
    base_path = merger.save_merged_export(merged_df, os.path.join(output_dir, "merged_data"))
    
    # Step 4: Chronological analysis
    analyzer = ChronologicalAnalyzer(merged_df)
    
    # Analyze progression
    progression = analyzer.analyze_relationship_progression()
    
    # Find critical periods
    critical_periods = analyzer.find_critical_periods()
    
    # Analyze balance
    balance = analyzer.analyze_communication_balance()
    
    # Extract threads
    threads = analyzer.extract_conversation_threads()
    
    # Step 5: Generate comprehensive report
    report = {
        'merger_stats': {
            'files_processed': len(export_files),
            'total_messages': len(merged_df),
            'duplicates_removed': merger.duplicates_removed,
            'date_range': f"{merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}",
            'participants': merged_df['sender'].unique().tolist()
        },
        'chronological_analysis': {
            'progression': progression,
            'critical_periods': critical_periods,
            'balance': balance,
            'top_threads': [
                {
                    'session_id': t['session_id'],
                    'start': t['start'].isoformat(),
                    'messages': t['messages'],
                    'duration_minutes': t['duration_minutes']
                }
                for t in threads[:10]
            ]
        }
    }
    
    # Save report
    report_path = os.path.join(output_dir, "chronological_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"  • Merged data: {base_path}.pkl")
    print(f"  • Analysis report: chronological_analysis_report.json")
    
    return merged_df, report


def main():
    """Main execution with CLI"""
    parser = argparse.ArgumentParser(
        description='Merge and analyze multiple WhatsApp exports chronologically'
    )
    parser.add_argument(
        'exports',
        nargs='+',
        help='WhatsApp export files to merge (e.g., chat1.txt chat2.txt chat3.txt)'
    )
    parser.add_argument(
        '--output',
        default='merged_analysis',
        help='Output directory for results (default: merged_analysis)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode - process only first 10000 messages from each export'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    valid_files = []
    for export_file in args.exports:
        if os.path.exists(export_file):
            valid_files.append(export_file)
            print(f"✅ Found: {export_file}")
        else:
            print(f"⚠️ Not found: {export_file}")
    
    if not valid_files:
        print("❌ No valid export files found")
        return
    
    print(f"\n📁 Processing {len(valid_files)} export files:")
    for f in valid_files:
        file_size = os.path.getsize(f) / (1024 * 1024)  # MB
        print(f"  • {Path(f).name} ({file_size:.1f} MB)")
    
    # Run pipeline
    try:
        merged_df, report = create_merged_analyzer_pipeline(valid_files, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 FINAL SUMMARY")
        print("="*60)
        print(f"Total messages merged: {report['merger_stats']['total_messages']:,}")
        print(f"Date range: {report['merger_stats']['date_range']}")
        print(f"Participants: {', '.join(report['merger_stats']['participants'])}")
        print(f"Duplicates removed: {report['merger_stats']['duplicates_removed']:,}")
        print(f"Conversation sessions: {len(report['chronological_analysis']['progression'])}")
        
        if report['chronological_analysis']['balance']:
            balance = report['chronological_analysis']['balance']
            if 'balance_ratio' in balance:
                print(f"Communication balance ratio: {balance['balance_ratio']:.2f}")
        
        print("\n✨ Your merged and chronologically organized data is ready for analysis!")
        print(f"📂 Check the {args.output}/ directory for all files")
        
        # Now ready for relationship analysis
        print("\n🧠 Ready for relationship analysis!")
        print(f"Run: python advanced_relationship_analyzer.py {args.output}/merged_data.pkl")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
