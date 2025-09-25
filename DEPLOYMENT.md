# 🚀 Deployment Guide for Render

## The Error You Encountered

The setuptools/build system error with complex dependencies. Here's the SIMPLE solution:

## ✅ Solution Files Created - SIMPLE VERSION

1. **`.python-version`** - Forces Python 3.10.8 (most stable)
2. **`runtime.txt`** - Python 3.10.8
3. **`render.yaml`** - Updated for simple version
4. **`Procfile`** - Uses main_simple.py
5. **`requirements-minimal.txt`** - ULTRA lightweight dependencies
6. **`main_simple.py`** - Simplified FastAPI app
7. **`model_trainer_simple.py`** - No pandas, no NLTK, no complex deps

## 🛠️ Render Deployment Steps

### Method 1: Using render.yaml (Recommended)

1. **Push your code to GitHub** with these files:
   - `.python-version`
   - `render.yaml`
   - `requirements.txt` (updated)
   - All your project files

2. **In Render Dashboard:**
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`
   - Deploy!

### Method 2: Manual Configuration

1. **Create Web Service** in Render
2. **Connect Repository**
3. **Configure Settings:**
   ```
   Name: toxic-comment-classifier
   Environment: Python
   Python Version: 3.11.0
   Build Command: pip install -r requirements.txt
   Start Command: python -m uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

## 🔧 Alternative: Use Lightweight Requirements

If you still get errors, use `requirements-deploy.txt`:

```bash
# Rename the file
mv requirements-deploy.txt requirements.txt
```

## 📱 Environment Variables (Optional)

Set these in Render if needed:
```
PYTHON_VERSION=3.11.0
PORT=10000
```

## 🚨 Common Issues & Solutions

### 1. **Python Version Issues**
- ✅ Fixed: Using Python 3.11.0 instead of 3.13.7
- ✅ Fixed: Downgraded pandas to 2.0.3

### 2. **Missing Dataset**
- App works without `train.csv`
- Shows appropriate message to users
- Can be uploaded later for training

### 3. **Port Configuration**
- ✅ Uses `$PORT` environment variable
- ✅ Binds to `0.0.0.0` for external access

### 4. **Memory Issues**
- Consider upgrading to paid Render plan for model training
- Free tier has limited memory (512MB)

## 🎯 Expected Behavior

**✅ Should work:**
- Web interface loads
- Model status checking
- Text prediction (if models exist)

**⏳ May need dataset:**
- Model training (requires train.csv upload)

## 📞 If Still Issues

1. Check Render build logs for specific errors
2. Try the lightweight requirements file
3. Consider using Python 3.10.x if 3.11 has issues
4. Upgrade to paid Render plan for more resources

Your app should now deploy successfully! 🎉