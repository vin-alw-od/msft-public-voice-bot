using EchoBot;
using Microsoft.Extensions.Logging.Configuration;
using Microsoft.Extensions.Logging.EventLog;
using Azure.Extensions.AspNetCore.Configuration.Secrets;
using Azure.Identity;

IHost host = Host.CreateDefaultBuilder(args)
    .UseWindowsService(options =>
    {
        options.ServiceName = "Echo Bot Service";
    })
    .ConfigureAppConfiguration((context, config) =>
    {
        // Add Azure Key Vault configuration
        var keyVaultEndpoint = "https://acu1-tmagent-bot-d1-kv.vault.azure.net/";
        
        try
        {
            config.AddAzureKeyVault(
                new Uri(keyVaultEndpoint),
                new DefaultAzureCredential()
            );
        }
        catch (Exception ex)
        {
            // Log the error but don't crash - fallback to appsettings
            var logger = LoggerFactory.Create(builder => builder.AddConsole()).CreateLogger("Program");
            logger.LogWarning($"Failed to connect to Key Vault: {ex.Message}. Falling back to appsettings.json");
        }
    })
    .ConfigureServices(services =>
    {
        LoggerProviderOptions.RegisterProviderOptions<
            EventLogSettings, EventLogLoggerProvider>(services);

        services.AddSingleton<IBotHost, BotHost>();

        services.AddHostedService<EchoBotWorker>();
    })
    .Build();

await host.RunAsync();
