"""
Quick Gmail Setup for AI Posture Monitor
Configure your Gmail credentials for email alerts
"""

import json

def setup_gmail_credentials():
    """Setup Gmail credentials in email_config.json"""
    print("📧 AI POSTURE MONITOR - GMAIL SETUP")
    print("=" * 40)
    print("🎯 Goal: Configure Gmail SMTP for email alerts")
    
    print("\n📋 GMAIL APP PASSWORD SETUP:")
    print("1. Go to: myaccount.google.com")
    print("2. Security → 2-Step Verification (enable if not enabled)")
    print("3. Security → App passwords")
    print("4. Select app: Mail → Generate")
    print("5. Copy the 16-character app password")
    print("\n⚠️ IMPORTANT: Use app password, NOT your regular Gmail password!")
    
    # Get user input
    gmail_address = input("\n📧 Your Gmail address: ").strip()
    if not gmail_address:
        print("❌ Gmail address is required!")
        return False
    
    app_password = input("🔑 Gmail app password (16 characters): ").strip()
    if not app_password:
        print("❌ App password is required!")
        return False
    
    # Load current config
    try:
        with open('email_config.json', 'r') as f:
            config = json.load(f)
    except:
        config = {}
    
    # Update SMTP settings
    config['smtp'] = {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": gmail_address,
        "password": app_password,
        "use_tls": True
    }
    
    config['recipient_email'] = gmail_address
    config['email_method'] = "smtp"
    config['mock_mode'] = False
    
    # Save updated config
    with open('email_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Gmail credentials saved!")
    
    # Test the configuration
    print("\n🧪 Testing Gmail connection...")
    
    import smtplib
    from email.mime.text import MIMEText
    
    try:
        # Test SMTP connection
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_address, app_password)
        
        # Send test email
        msg = MIMEText("🎉 Gmail SMTP setup successful! Your AI Posture Monitor is ready to send email alerts.")
        msg['Subject'] = "🧪 AI Posture Monitor - Gmail Setup Test"
        msg['From'] = gmail_address
        msg['To'] = gmail_address
        
        server.send_message(msg)
        server.quit()
        
        print("✅ Test email sent successfully!")
        print(f"📬 Check your Gmail inbox: {gmail_address}")
        print("\n🎉 GMAIL SETUP COMPLETE!")
        print("✅ Email alerts are now enabled")
        print("🚨 Fall detection alerts will be sent to your Gmail")
        
        return True
        
    except Exception as e:
        print(f"❌ Gmail setup failed: {str(e)}")
        
        if "authentication failed" in str(e).lower():
            print("\n💡 TROUBLESHOOTING:")
            print("   • Make sure 2-factor authentication is enabled")
            print("   • Use app password, NOT regular password")
            print("   • App password should be 16 characters")
            print("   • Try generating a new app password")
        
        return False

def main():
    print("🏥 AI POSTURE MONITOR - GMAIL CONFIGURATION")
    print("🚫 Web3Forms completely removed")
    print("📧 Setting up Gmail SMTP for reliable email delivery")
    
    success = setup_gmail_credentials()
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run: python integrated_monitor.py")
        print("2. Start monitoring for falls")
        print("3. Receive instant email alerts!")
    else:
        print("\n🔄 Try setup again with correct credentials")

if __name__ == "__main__":
    main()
