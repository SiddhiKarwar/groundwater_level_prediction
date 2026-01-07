import pandas as pd

# Load the new dataset
df = pd.read_csv('cgwb_borewells_nashik.csv')

print("="*70)
print("✅ NEW BOREWELL DATASET VERIFICATION")
print("="*70)
print(f"\n🔢 Total Records: {len(df):,} borewells")
print(f"📊 Columns: {len(df.columns)} - {list(df.columns)}")
print(f"\n🏘️  Unique Talukas: {df['Taluka'].nunique()}")
print(f"   Talukas: {', '.join(sorted(df['Taluka'].unique()))}")

success_count = len(df[df['Status'] == 'Success'])
success_rate = (success_count / len(df)) * 100
print(f"\n✅ Success Rate: {success_rate:.2f}%")
print(f"   Successful: {success_count:,}")
print(f"   Failed: {len(df) - success_count:,}")

print(f"\n📍 Geographic Range:")
print(f"   Latitude:  {df['Latitude'].min():.4f} to {df['Latitude'].max():.4f}")
print(f"   Longitude: {df['Longitude'].min():.4f} to {df['Longitude'].max():.4f}")

print(f"\n🌊 Depth Range: {df['Depth_m'].min():.1f}m to {df['Depth_m'].max():.1f}m (avg: {df['Depth_m'].mean():.1f}m)")
print(f"💧 Yield Range: {df['Yield_LPH'].min():,} to {df['Yield_LPH'].max():,} LPH (avg: {df['Yield_LPH'].mean():,.0f} LPH)")

print(f"\n📅 Construction Years: {df['Construction_Year'].min()} to {df['Construction_Year'].max()}")

print("\n" + "="*70)
print("🎉 Dataset ready for use! Restart Flask app to see all borewells!")
print("="*70)
