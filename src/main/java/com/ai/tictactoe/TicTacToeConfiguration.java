package com.ai.tictactoe;

import com.ai.tictactoe.game.AnnTicTacToeAgent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.ViewResolver;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.ViewResolverRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.thymeleaf.spring5.SpringTemplateEngine;
import org.thymeleaf.spring5.templateresolver.SpringResourceTemplateResolver;
import org.thymeleaf.spring5.view.ThymeleafViewResolver;

@Configuration
@EnableWebMvc
@ComponentScan(basePackages = {"com.ai.tictactoe"})
public class TicTacToeConfiguration implements WebMvcConfigurer
{
    @Autowired
    private ApplicationContext applicationContext;

    @Bean
    AnnTicTacToeAgent annTicTacToeAgent()
    {
        AnnTicTacToeAgent ann =  new AnnTicTacToeAgent("x");
        //ann.init("net-18-15-12-9-20210120-0840-batch-size-630-epochs-1418.ann");
        ann.init("net-27-36-9-20210119-0853-batch-size-630-epochs-1353.ann");
        return ann;
    }


    @Bean
    public SpringResourceTemplateResolver templateResolver()
    {
        SpringResourceTemplateResolver templateResolver = new SpringResourceTemplateResolver();
        templateResolver.setApplicationContext(applicationContext);
        templateResolver.setPrefix("/WEB-INF/templates/");
        templateResolver.setSuffix(".html");
        return templateResolver;
    }

    @Bean
    public SpringTemplateEngine templateEngine()
    {
        SpringTemplateEngine templateEngine = new SpringTemplateEngine();
        templateEngine.setTemplateResolver(templateResolver());
        templateEngine.setEnableSpringELCompiler(true);
        return templateEngine;
    }

    @Bean
    public ViewResolver viewResolver()
    {
        ThymeleafViewResolver resolver = new ThymeleafViewResolver();
        ViewResolverRegistry registry = new ViewResolverRegistry(null, applicationContext);
        resolver.setTemplateEngine(templateEngine());
        registry.viewResolver(resolver);
        return resolver;
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry)
    {
        registry.addResourceHandler("/js/**").addResourceLocations("classpath:/static/js/");
        registry.addResourceHandler("/css/**").addResourceLocations("classpath:/static/css/");
        registry.addResourceHandler("/images/**").addResourceLocations("classpath:/static/images/");
    }
}
